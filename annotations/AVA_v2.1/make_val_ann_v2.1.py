import argparse
import csv
import os
import json
import copy
import pickle as pkl
from tqdm import tqdm


def main(args):
   
    with open(args.ava_v2_path, 'rb') as f:
        ava_v2_data = pkl.load(f)
    class_dict = ava_v2_data[1]

    class_to_idx = {}
    for i, class_label in enumerate(class_dict):
        id = class_label['id']
        class_to_idx[id] = i

    csv_file = open(args.data_path, "r", encoding="ms932", errors="", newline="" )
    f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
    
    csv_lists = []
    for row in f:
        csv_lists.append(row)

    total_clip_list = []
    clip_list = []
    for i, row in tqdm(enumerate(csv_lists)):
        if len(row) < 7:
            continue
        
        current_video = row[0]
        current_time = int(float(row[1]))
        
        if len(clip_list) == 0:
            clip_list.append(row)
            pre_video = current_video
            pre_time = current_time
        
        elif current_video == pre_video and current_time == pre_time:
            clip_list.append(row)
            pre_video = current_video
            pre_time = current_time

        else:
            total_clip_list.append(clip_list)
            pre_video = current_video
            pre_time = current_time
            clip_list = [row]

    clips_info = []
    for clip_list in tqdm(total_clip_list):
        if len(clip_list[0]) < 7:
            continue

        video_name = clip_list[0][0]
        time = int(float(clip_list[0][1]))
        start_time = 900

        clip_dict = {}
        clip_dict['video'] = video_name
        clip_dict['time'] = time
        clip_dict['mid_frame'] = int((time - start_time) * 30 + 1)
        clip_dict['start_frame'] = max(int(clip_dict['mid_frame'] - 45), 1)
        clip_dict['n_frames'] = int(91)
        clip_dict['format_str'] = 'image_%06d.jpg'
        clip_dict['frame_rate'] = 30.0

        label_list = []
        for clip in clip_list:
            # only use ava action labels
            if int(float(clip[6])) in class_to_idx.keys():        
                bounding_box = [float(clip[2]), float(clip[3]), float(clip[4]), float(clip[5])]  
                label = class_to_idx[int(float(clip[6]))]
                lable_set = {'bounding_box': bounding_box, 'label': [label]}
            else:
                bounding_box = [float(clip[2]), float(clip[3]), float(clip[4]), float(clip[5])]  
                label = 60
                lable_set = {'bounding_box': bounding_box, 'label': [label]}

            if len(label_list) == 0:
                label_list.append(lable_set)
            else:
                for labels in label_list:
                    if labels['bounding_box'] == bounding_box:
                        if 60 in labels['label'] and label == 60:
                            new_bbox = False
                            continue
  
                        labels['label'].append(label)
                        labels['label'].sort()
                        new_bbox = False
                    else:
                        new_bbox = True
                
                if new_bbox is True:
                    label_list.append(lable_set)
        
        for labels in label_list:
            if any(x < 13 for x in labels['label']) and (60 in labels['label']):
                labels['label'].remove(60)

        clip_dict['labels'] = label_list
        clips_info.append(clip_dict)

    print(len(clips_info))
    with open(args.out_path, 'wb') as f:
        pkl.dump((clips_info, class_dict), f, protocol=pkl.HIGHEST_PROTOCOL)
    
    # total_clips_info = clips_info + ava_v2_data[0]
    # with open(args.total_out_path, 'wb') as f:
    #     pkl.dump((total_clips_info, class_dict), f, protocol=pkl.HIGHEST_PROTOCOL)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Port action dataset')
    parser.add_argument('--data_path', type=str, required=False,
                        help='data file', default="./ava_val_v2.1.csv")
    parser.add_argument('--ava_v2_path', type=str, required=False,
                        help='data file', default="../AVA_v2.2/ava_train_v2.2.pkl")          
    parser.add_argument('--out_path', type=str, required=False,
                        help='data file', default="./ava_val_v2.1_all.pkl")                           
    args = parser.parse_args()    
    
    main(args)
