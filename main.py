import sys
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import os
import argparse
import json
import pprint
import socket
import time
from easydict import EasyDict
import yaml
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torchvision.ops.boxes import box_area

from calc_mAP import run_evaluation, run_det_evaluation
from datasets import ava, spatial_transforms, temporal_transforms
from distributed_utils import init_distributed
from models import JARViS
from detr.models import build_model as build_detector
from models.preprocessor import Preprocessor, get_bbox_after_aug

from scheduler import get_scheduler
from utils import *
from util import box_ops


def main(local_rank, args):
    '''dist init'''
    rank, world_size = init_distributed(local_rank, args)

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    opt = EasyDict(config)
    opt.world_size = world_size

    if rank == 0:
        mkdir(opt.result_path)
        mkdir(os.path.join(opt.result_path, 'tmp'))
        with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file, indent=2)
        logger = create_logger(os.path.join(opt.result_path, 'log.txt'))
        logger.info('opt: {}'.format(pprint.pformat(opt, indent=2)))
        
        writer = SummaryWriter(os.path.join(opt.result_path, 'tb'))
    else:
        logger = writer = None
    dist.barrier()

    random_seed(opt.manual_seed)
    # setting benchmark to True causes OOM in some cases
    if opt.get('cudnn', None) is not None:
        torch.backends.cudnn.deterministic = opt.cudnn.get('deterministic', False)
        torch.backends.cudnn.benchmark = opt.cudnn.get('benchmark', False)

    # create model
    net = JARViS(opt.model)
    net.cuda()
    net = DistributedDataParallel(net, device_ids=[local_rank], broadcast_buffers=False)

    # create human detection model
    detector, _, _ = build_detector(opt.detector)

    checkpoint = torch.load(opt.detector.pretrained, map_location='cpu')
    detector.load_state_dict(checkpoint)
    detector.cuda()

    preprocessor = Preprocessor(opt.preprocessor)

    if rank == 0:
        logger.info(net)
        logger.info(parameters_string(net))

    if not opt.get('evaluate', False):
        train_aug = opt.train.augmentation

        spatial_transform = [getattr(spatial_transforms, aug.type)(**aug.get('kwargs', {})) for aug in train_aug.spatial]
        spatial_transform = spatial_transforms.Compose(spatial_transform)
    
        temporal_transform = getattr(temporal_transforms, train_aug.temporal.type)(**train_aug.temporal.get('kwargs', {}))

        det_spatial_transform = [getattr(spatial_transforms, aug.type)(**aug.get('kwargs', {})) for aug in train_aug.det_spatial]
        det_spatial_transform = spatial_transforms.Compose(det_spatial_transform)

        train_data = ava.AVA(
            opt.train.root_path,
            opt.train.annotation_path,
            spatial_transform,
            temporal_transform,
            det_spatial_transform
        )

        train_sampler = DistributedSampler(train_data, round_down=True)

        train_loader = ava.AVADataLoader(
            train_data,
            batch_size=opt.train.batch_size,
            shuffle=False,
            num_workers=opt.train.get('workers', 1),
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True
        )

        if rank == 0:
            logger.info('# train data: {}'.format(len(train_data)))
            logger.info('train spatial aug: {}'.format(spatial_transform))
            logger.info('train temporal aug: {}'.format(temporal_transform))

            train_logger = Logger(
                os.path.join(opt.result_path, 'train.log'),
                ['epoch', 'loss', 'lr'])
            train_batch_logger = Logger(
                os.path.join(opt.result_path, 'train_batch.log'),
                ['epoch', 'batch', 'iter', 'loss', 'lr'])
        else:
            train_logger = train_batch_logger = None

        optim_opt = opt.train.optimizer
        sched_opt = opt.train.scheduler

        net_without_ddp = net.module
        param_dicts = [
            {"params": [p for n, p in net_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": optim_opt.lr_backbone,
            },
        ]

        optimizer = torch.optim.AdamW(param_dicts, lr=optim_opt.lr,
                                        weight_decay=optim_opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, sched_opt.lr_drop)

    val_aug = opt.val.augmentation

    spatial_transform = [getattr(spatial_transforms, aug.type)(**aug.get('kwargs', {})) for aug in val_aug.spatial]
    spatial_transform = spatial_transforms.Compose(spatial_transform)
    
    temporal_transform = getattr(temporal_transforms, val_aug.temporal.type)(**val_aug.temporal.get('kwargs', {}))

    det_spatial_transform = [getattr(spatial_transforms, aug.type)(**aug.get('kwargs', {})) for aug in val_aug.det_spatial]
    det_spatial_transform = spatial_transforms.Compose(det_spatial_transform)

    val_data = ava.AVA(
        opt.val.root_path,
        opt.val.annotation_path,
        spatial_transform,
        temporal_transform,
        det_spatial_transform
    )

    val_sampler = DistributedSampler(val_data, round_down=False)

    val_loader = ava.AVADataLoader(
        val_data,
        batch_size=opt.val.batch_size,
        shuffle=False,
        num_workers=opt.val.get('workers', 1),
        pin_memory=True,
        sampler=val_sampler
    )

    val_logger = None
    if rank == 0:
        logger.info('# val data: {}'.format(len(val_data)))
        logger.info('val spatial aug: {}'.format(spatial_transform))
        logger.info('val temporal aug: {}'.format(temporal_transform))

        val_log_items = ['epoch']
        if opt.val.with_label:
            val_log_items.append('loss')
        if opt.val.get('eval_mAP', None) is not None:
            val_log_items.append('mAP')
        if len(val_log_items) > 1:
            val_logger = Logger(
                os.path.join(opt.result_path, 'val.log'),
                val_log_items)

    if opt.get('pretrain', None) is not None:
        load_pretrain(opt.pretrain, net)

    begin_epoch = 1
    if opt.get('resume_path', None) is not None:
        if not os.path.isfile(opt.resume_path):
            opt.resume_path = os.path.join(opt.result_path, opt.resume_path)
        checkpoint = torch.load(opt.resume_path, map_location=lambda storage, loc: storage.cuda())

        begin_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['state_dict'])
        if rank == 0:
            logger.info('Resumed from checkpoint {}'.format(opt.resume_path))

        if not opt.get('evaluate', False):
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, sched_opt.lr_drop)
            # optimizer.zero_grad()
            # optimizer.step()           
            # scheduler.step()     

            if rank == 0:
                logger.info('Also loaded optimizer and scheduler from checkpoint {}'.format(opt.resume_path))
        del checkpoint

    if opt.get('evaluate', False):  
        # evaluation mode
        val_epoch(begin_epoch - 1, val_loader, net, detector, preprocessor, 
                  opt, logger, val_logger, rank, world_size, writer)
    else:  
        # training and validation mode
        for e in range(begin_epoch, opt.train.n_epochs + 1):
            train_sampler.set_epoch(e)
            train_epoch(e, train_loader, net, detector, preprocessor, optimizer, scheduler,
                        opt, logger, train_logger, train_batch_logger, rank, world_size, writer)
            scheduler.step()
            if e % opt.train.val_freq == 0 or e == opt.train.n_epochs:
                val_epoch(e, val_loader, net, detector, preprocessor,
                          opt, logger, val_logger, rank, world_size, writer)

    if rank == 0:
        writer.close()


def train_epoch(epoch, data_loader, model, detector, preprocessor, optimizer, scheduler,
                opt, logger, epoch_logger, batch_logger, rank, world_size, writer):
    if rank == 0:
        logger.info('Training at epoch {}'.format(epoch))

    model.train()
    detector.eval()

    batch_time = AverageMeter(opt.print_freq)
    data_time = AverageMeter(opt.print_freq)
    loss_time = AverageMeter(opt.print_freq)
    losses = AverageMeter(opt.print_freq)
    class_error = AverageMeter(opt.print_freq)
    global_losses = AverageMeter()

    accum_iter = opt.accum_iter

    end_time = time.time()
    for i, data in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        i_p = {'aug_info': data['aug_info'], 'labels': data['labels'],
               'filenames': data['filenames'], 'mid_times': data['mid_times']}        
        o_p = preprocessor(i_p)

        with torch.no_grad():
            det_inputs = data['images'].to('cuda')
            outputs = detector(det_inputs)

            prob = F.softmax(outputs['pred_logits'], -1)[..., 1]
            out_bbox = outputs['pred_boxes']
            out_hs = outputs['hidden_states']    

            pred_scores, topk_indexes = torch.topk(prob, opt.model.topk, dim=1)
            
            # convert to [x0, y0, x1, y1] format
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            pred_boxes = torch.gather(boxes, 1, topk_indexes.unsqueeze(-1).repeat(1,1,4))
            roi_features = torch.gather(out_hs, 1, topk_indexes.unsqueeze(-1).repeat(1,1,out_hs.shape[-1]))

            inputs = {'clips': data['clips'].cuda(), 'sizes_before_padding': o_p['sizes_before_padding']}
            inputs['pred_scores'] = pred_scores.unsqueeze(-1)
            inputs['pred_boxes'] = pred_boxes
            inputs['roi_features'] = roi_features
            inputs['pred_boxes_mask'] = None

        ret = model(inputs)
        outputs = ret['outputs']
        criterion = ret['criterion']

        loss_dict = criterion(outputs, o_p['targets'])
        loss = loss_dict['loss'] / accum_iter
        cls_error = loss_dict['class_error']

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            print(loss_dict)
            sys.exit(1)

        loss.backward()

        # weights update
        if ((i + 1) % accum_iter == 0) or (i + 1 == len(data_loader)):
            if opt.train.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.train.max_norm)        
            optimizer.step()
            optimizer.zero_grad()

        reduced_loss = loss.clone()
        dist.all_reduce(reduced_loss)
        losses.update(reduced_loss.item(), world_size)
        global_losses.update(reduced_loss.item(), world_size)

        reduced_class_error = cls_error.clone()
        dist.all_reduce(reduced_class_error)
        reduced_class_error = reduced_class_error /  world_size
        class_error.update(reduced_class_error.item())

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        curr_step = (epoch - 1) * len(data_loader) + i
        if (i + 1) % opt.print_freq == 0 and rank == 0:
            writer.add_scalar('train/loss', losses.avg, curr_step + 1)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], curr_step + 1)

            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': curr_step + 1,
                'loss': losses.avg,
                'lr': optimizer.param_groups[0]['lr']
            })

            logger.info('Epoch [{0}]\t'
                        'Iter [{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss1.val:.4f} ({loss1.avg:.4f})\t'
                        'Class error {error1.val:.3f} ({error1.avg:.3f})'.format(
                                                                                epoch,
                                                                                i + 1,
                                                                                len(data_loader),
                                                                                batch_time=batch_time,
                                                                                data_time=data_time,
                                                                                loss1=losses,
                                                                                error1=class_error))

    if rank == 0:
        writer.add_scalar('train/epoch_loss', global_losses.avg, epoch)
        writer.flush()

        epoch_logger.log({
            'epoch': epoch,
            'loss': global_losses.avg,
            'lr': optimizer.param_groups[0]['lr']
        })

        logger.info('-' * 100)
        logger.info(
            'Epoch [{}/{}]\t'
            'Loss {:.4f}'.format(
                epoch,
                opt.train.n_epochs,
                global_losses.avg))

        if epoch % opt.train.save_freq == 0:
            save_file_path = os.path.join(opt.result_path, 'ckpt_{}.pth.tar'.format(epoch))
            states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(states, save_file_path)
            logger.info('Checkpoint saved to {}'.format(save_file_path))

        logger.info('-' * 100)


def val_epoch(epoch, data_loader, model, detector, preprocessor,
              opt, logger, epoch_logger, rank, world_size, writer):
    if rank == 0:
        logger.info('Evaluation at epoch {}'.format(epoch))

    model.eval()
    detector.eval()
    
    calc_loss = opt.val.with_label
    out_file = open(os.path.join(opt.result_path, 'tmp', 'predict_rank%d.csv'%rank), 'w')

    batch_time = AverageMeter(opt.print_freq)
    data_time = AverageMeter(opt.print_freq)

    end_time = time.time()
    for i, data in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        i_p = {'aug_info': data['aug_info'], 'labels': data['labels'],
               'filenames': data['filenames'], 'mid_times': data['mid_times']}        
        o_p = preprocessor(i_p)

        with torch.no_grad():
            det_inputs = data['images'].to('cuda')
            outputs = detector(det_inputs)

            prob = F.softmax(outputs['pred_logits'], -1)[..., 1]
            out_bbox = outputs['pred_boxes']
            out_hs = outputs['hidden_states']    

            pred_scores, topk_indexes = torch.topk(prob, opt.model.topk, dim=1)
            
            # convert to [x0, y0, x1, y1] format
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            pred_boxes = torch.gather(boxes, 1, topk_indexes.unsqueeze(-1).repeat(1,1,4))
            roi_features = torch.gather(out_hs, 1, topk_indexes.unsqueeze(-1).repeat(1,1,out_hs.shape[-1]))

            inputs = {'clips': data['clips'].cuda(), 'sizes_before_padding': o_p['sizes_before_padding']}
            inputs['pred_scores'] = pred_scores.unsqueeze(-1)
            inputs['pred_boxes'] = pred_boxes
            inputs['roi_features'] = roi_features
            inputs['pred_boxes_mask'] = None

            ret = model(inputs)
            outputs = ret['outputs']
            postprocessor = ret['postprocessor']

        results = postprocessor['action'](outputs)

        fnames, mid_times = data['filenames'], data['mid_times']
        idx_to_class = data_loader.dataset.idx_to_class
        for batch_idx in range(len(results)):
            for obj_idx in range(len(results[batch_idx]['boxes'])):
                output_box = results[batch_idx]['boxes'][obj_idx].cpu()

                prefix = "%s,%s,%.3f,%.3f,%.3f,%.3f"%(fnames[batch_idx], mid_times[batch_idx],
                                                        output_box[0], output_box[1],
                                                        output_box[2], output_box[3])
                cls = results[batch_idx]['labels'][obj_idx]
                score_str = '%.3f'%results[batch_idx]['scores'][obj_idx]
                out_file.write(prefix + ",%d,%s\n" % (idx_to_class[cls]['id'], score_str))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (i + 1) % opt.print_freq == 0 and rank == 0:
            logger.info('Epoch [{0}]\t'
                        'Iter [{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                            epoch,
                            i + 1,
                            len(data_loader),
                            batch_time=batch_time,
                            data_time=data_time))

    out_file.close()
    dist.barrier()

    if rank == 0:
        val_log = {'epoch': epoch}
        val_str = 'Epoch [{}]'.format(epoch)

        result_file = os.path.join(opt.result_path, 'predict_epoch%d.csv'%epoch)
        with open(result_file, 'w') as of:
            for r in range(world_size):
                with open(os.path.join(opt.result_path, 'tmp', 'predict_rank%d.csv'%r), 'r') as f:
                    of.writelines(f.readlines())

        if opt.val.get('eval_mAP', None) is not None:
            eval_mAP = opt.val.eval_mAP
            metrics = run_evaluation(
                open(eval_mAP.labelmap, 'r'), 
                open(eval_mAP.groundtruth, 'r'),
                open(result_file, 'r'),
                open(eval_mAP.exclusions, 'r') if eval_mAP.get('exclusions', None) is not None else None, 
                logger
            )
            
            mAP = metrics['PascalBoxes_Precision/mAP@0.5IOU']
            writer.add_scalar('val/mAP', mAP, epoch)

            val_log['mAP'] = mAP
            val_str += '\tmAP {:.6f}'.format(mAP)

        writer.flush()

        if epoch_logger is not None:
            epoch_logger.log(val_log)

            logger.info('-' * 100)
            logger.info(val_str)
            logger.info('-' * 100)

    dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch AVA Training and Evaluation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--nproc_per_node', type=int, default=1)
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--master_addr', type=str, default=socket.gethostbyname(socket.gethostname()))
    parser.add_argument('--master_port', type=int, default=31114)
    parser.add_argument('--nnodes', type=int, default=None)
    parser.add_argument('--node_rank', type=int, default=None)
    args = parser.parse_args()

    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.nproc_per_node)


