import os
import json
import numpy as np
import cv2
from lxml import etree

def load_json(path):
    with open(path,'r') as load_f:
        dict_4_json = json.load(load_f)
    return dict_4_json

def find_data_path(path="./kuma"):
    videos = []
    annos = []
    for f in os.walk(path):
        for fname in f[2]:
            if fname[-4:]==".mp4":
                videos.append(os.path.join(f[0],fname))
            elif fname == "annotations.json":
                annos.append(os.path.join(f[0],fname))
    print(videos, annos)
    return videos, annos


def get_box(path, mode="xml"):
    boxes = []
    if mode == "xml":
        root = etree.parse(path).getroot()
        for e in root.iter():
            if e.tag == 'box':
                boxes.append(e.attrib)
    elif mode == "json":
        anno = load_json(path=path)
        tracks = anno[0]['tracks'][0]
        boxes.append(tracks["shapes"])

    return boxes

def make_dataset(files):
    os.system("rm -rf ../data/kuma")
    os.system("mkdir -p ../data/kuma/crop511/train")
    os.system("mkdir -p ../data/kuma/crop511/valid")
    
    for idx,f in enumerate(files):
        cap_file = cv2.VideoCapture(f)
        success, frame = cap_file.read()
        count = 0
        while success:
            os.system("mkdir -p ../data/kuma/crop511/train/KUMA_{}".format(idx))
            frame_num = "{:06d}".format(count)
            cv2.imwrite("../data/kuma/crop511/train/KUMA_{}/{}.00.x.jpg".format(idx, frame_num), frame)     # save frame as JPEG file
            success, frame = cap_file.read()
            count += 1
        cap_file.release()
        
def draw(image, box, name="test.jpg"):
    """
    draw image for debugging
    """
    draw_image = np.array(image.copy())
    x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
    cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0, 255, 0))
    cv2.circle(draw_image, (int(round(x1 + x2) / 2), int(round(y1 + y2) / 2)), 3, (0, 0, 255))
    cv2.putText(draw_image, '[x: {}, y: {}]'.format(int(round(x1 + x2) / 2), int(round(y1 + y2) / 2)),
                (int(round(x1 + x2) / 2) - 3, int(round(y1 + y2) / 2) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                (255, 255, 255), 1)
    cv2.imwrite(name, draw_image) 

def gen_json_labels(path="../data/kuma/", label_paths=["./annotations.xml"], name="all.json"):
    json_path = os.path.join(path, name)
    box_labels = [get_box(p, mode="json") for p in label_paths]
    
    all_info = {}
    for idx_v, anno in enumerate(box_labels):
        all_info["train/KUMA_"+str(idx_v)] = {}
        all_info["train/KUMA_"+str(idx_v)]["00"] = {}
        for label in anno[0]:
            rect = [int(pos) for pos in label["points"]]
            frame = "{:06d}".format(label["frame"])
            all_info["train/KUMA_"+str(idx_v)]["00"][frame] = rect


    # all_info = {}
    # train_video_path = os.path.join(path, "crop511/train/")
    # train_videos = os.listdir(train_video_path)

    # for idx_v,t_v in enumerate(train_videos):
    #     all_info["train/KUMA_"+str(idx_v)] = {}
    #     all_info["train/KUMA_"+str(idx_v)]["00"] = {}
    #     train_figs = os.listdir(os.path.join(train_video_path, t_v))
    #     for idx_f in range(len(train_figs)):
    #         label = [int(a) for a in box_labels[idx_v][idx_f]]
    #         frame = "{:06d}".format(idx_f)
    #         all_info["train/KUMA_"+str(idx_v)]["00"][frame] = label

    with open(json_path,"w") as f:
        json.dump(all_info, f)

if __name__ == "__main__":
    
    videos,annos = find_data_path()
    # make_dataset(videos)
    gen_json_labels(path="../data/kuma/", label_paths=annos)
    # load_json("../data/kuma/all.json")