import cv2
import os
import json
import numpy as np

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

def read_result(path='../result/KUMA/', name='AutoMatchcheckpoint_e20/kuma.txt'):
    result = np.loadtxt(os.path.join(path,name))
    return result

def draw_video(path='../dataset/KUMA/'):
    json_path = os.path.join(path,'test.json')
    videos = json.load(open(json_path,'r'))['image_files']
    pred = read_result()
    for id, fig in enumerate(videos):
        import pdb
        pdb.set_trace()
        p = [int(ax) for ax in pred[id]]
        draw(fig,p,name="./drawout/{}.jpg".format(id))

if __name__ == "__main__":
    draw_video()



