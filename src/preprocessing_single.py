import cv2
import os
import time
import argparse
from tqdm import tqdm


def time_func(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print("-"*30)
        print(f"Runtime: {end-start:0.2f}s")
    return wrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--training_dir', help='How many captchas to generate', type=str, required=True)
    parser.add_argument('-o','--output_dir', help='Where to store the generated captchas', type=str, required=True)
    args = parser.parse_args()
    return args


def clean_img(img):
    mod_img = img.copy()
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey,128,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #first applying MSER on small areas to remove noise
    mser = cv2.MSER().create()
    mser.setMaxArea(30)
    mser.setMinArea(1)
    _, rects = mser.detectRegions(thresh)
    for (x,y,w,h) in rects:
        cv2.rectangle(img, (x,y), (x+w,y+h), color=(0,0,0), thickness=1)
        mod_img[y:y+h,x:x+w,:].fill(255)

    grey = cv2.cvtColor(mod_img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(5,5),0)
    _, thresh = cv2.threshold(blur,128,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    #next applying MSER on large areas to select characters
    mser = cv2.MSER().create()
    mser.setMaxArea(mod_img.shape[0]*mod_img.shape[1]//7)
    mser.setMinArea(150)
    _, rects = mser.detectRegions(thresh)
    for (x,y,w,h) in rects:
        cv2.rectangle(mod_img, (x,y), (x+w,y+h), color=(0,0,0), thickness=1)
    #can crop by bounding box/padded bounding box now
    
    return thresh


@time_func
def main():
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    for img_dir in tqdm(os.listdir(args.training_dir)):
        dir_path = os.path.join(args.training_dir, img_dir)
        out_path = os.path.join(args.output_dir, img_dir)
        os.makedirs(out_path, exist_ok=True)
        for img in os.listdir(dir_path):
            img_data = cv2.imread(os.path.join(dir_path, img))
            img_data = clean_img(img_data)
            cv2.imwrite(os.path.join(out_path, img), img_data)



if __name__ == "__main__":
    main()