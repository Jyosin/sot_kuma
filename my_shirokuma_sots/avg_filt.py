import cv2
import os
import json
import numpy as np


def draw(image, avg, name="test.jpg"):
    draw_image = np.array(image.copy())
    filted_img = draw_image - avg
    cv2.imwrite(name, filted_img) 

def cal_avg(path, videos, win_size=20):
    win = []
    for id, fig in enumerate(videos):
        image = cv2.imread(os.path.join(path,fig)) 
        win.append(image)
    # import pdb
    # pdb.set_trace()
    avg = np.sum(win,0)/len(win)
    return avg

def draw_video(path='../dataset/kuma/', json_name="test.json"):
    json_path = os.path.join(path, json_name)
    videos = json.load(open(json_path,'r'))['image_files']
    avg = cal_avg(path, videos)
    
    for id, fig in enumerate(videos):
        image = cv2.imread(os.path.join(path,fig)) 
        draw(image, avg, name="test_{}.jpg".format(id))
        
if __name__ == "__main__":
    draw_video()