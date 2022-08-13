import os
import json
import numpy as np
import cv2
from lxml import etree

path = "./annotations.xml"
videos = ["../data/polar/91_2020_09_22_08.mp4","../data/polar/91_2020_09_22_08.mp4"]

def get_box(path):
    boxes = []
    root = etree.parse(path).getroot()
    for e in root.iter():
        if e.tag == 'box':
            boxes.append(e.attrib)
    return boxes

def make_dataset(files):
    os.system("rm -rf ../dataset/kuma")
    os.system("mkdir -p ../dataset/kuma/test")

    for idx,f in enumerate(files):
        os.system("mkdir -p ../dataset/kuma/test/KUMA_{}".format(idx))
        cap_file = cv2.VideoCapture(f)
        success, frame = cap_file.read()
        count = 0
        while success:
            frame_num = "{:06d}".format(count)
            cv2.imwrite("../dataset/kuma/test/KUMA_{}/{}.jpg".format(idx, frame_num), frame)     # save frame as JPEG file
            success, frame = cap_file.read()
            count += 1
        cap_file.release()
    print("Test images set build finished!")

def gen_json_labels(path="../dataset/kuma", label_path="./annotations.xml", name="test.json"):
    json_path = os.path.join(path, name)
    
    box_labels = get_box(label_path)
    box_args = ['xtl','ytl','xbr','ybr']
    
    all_info = {"gt_rect":[],"init_rect":[],"image_files":[],"gt":[]}
    
    test_video_path = os.path.join(path, "test/")
    
    test_videos = os.listdir(test_video_path)

    for idx_v,t_v in enumerate(test_videos):
        test_figs = os.listdir(os.path.join(test_videos, t_v))
        all_info += test_figs
        for idx_f in range(len(test_figs)):
            label = [int(np.float32(box_labels[idx_f][a])) for a in box_args]
            if idx_f == 0:
                all_info["init_rect"] = label
            all_info["gt_rect"].append(label)
    all_info["gt"]=np.array(all_info["gt_rect"])
                
    with open(json_path,"w") as f:
        json.dump(all_info, f)

def load_json(path):
    with open(path,'r') as load_f:
        dict_4_json = json.load(load_f)
    return dict_4_json

if __name__ == "__main__":
    # index = [1791,1891,1991,2001,2011,2021,2031]
    # for idx in index:
    #     frame_num = "{:06d}".format(idx)
    #     image_name = "../data/kuma/crop511/train/KUMA_0/{}.00.x.jpg".format(frame_num)
    #     image = cv2.imread(image_name)
    #     b = get_box(path)
    #     box_args = ['xtl','ytl','xbr','ybr']
    #     box = [np.float32(b[idx][a]) for a in box_args]
    #     draw(image, box,name="./test_{}.jpg".format(idx))
    # make_dataset(videos)
    gen_json_labels()
    # make_dataset(videos)
    # gen_json_labels()
    # load_json("../data/kuma/all.json")S