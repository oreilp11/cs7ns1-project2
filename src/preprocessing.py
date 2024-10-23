import cv2
import os
import time
import random
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
    parser.add_argument('-i','--image_dir', help='How many captchas to generate', type=str, required=True)
    parser.add_argument('-o','--output-dir', help='Where to store the generated captchas', type=str, required=True)
    parser.add_argument('-c','--count', help='How many images to test', type=int, required=False, default=50)
    args = parser.parse_args()
    return args


@time_func
def main():
    args = parse_args()
    imgs = sorted(os.listdir(args.image_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    for _ in tqdm(range(args.count)):
        img_name = random.choice(imgs)
        img = cv2.imread(os.path.join(args.image_dir, img_name))
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

        cv2.imwrite(os.path.join(args.output_dir,f'bbox_{img_name}'),mod_img)
        cv2.imwrite(os.path.join(args.output_dir,f'thresh_{img_name}'),thresh)


if __name__ == "__main__":
    main()