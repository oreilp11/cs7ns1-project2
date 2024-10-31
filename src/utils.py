import time
import cv2
import numpy as np

def time_func(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print("-"*30)
        print(f"Runtime: {end-start:0.2f}s")

    return wrapper

def clean_img(img):
    kernel = np.ones((3,3)) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(3,3),0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mser = cv2.MSER().create()
    mser.setMaxArea(20)
    mser.setMinArea(1)
    _, rects = mser.detectRegions(img)
    for x,y,w,h in rects:
        img[y:y+h,x:x+w] = 255
    
    img = cv2.dilate(img, kernel)
    img = cv2.medianBlur(img, 3)
    img = cv2.erode(img, kernel)

    return img

def split_img(img):
    mser = cv2.MSER().create()
    mser.setMaxArea(img.shape[0]*img.shape[1]//7)
    mser.setMinArea(150)
    _, rects = mser.detectRegions(img)

    return [img[y:y+h,x:x+w] for x,y,w,h in rects]


def center_resize(img, width=96, height=96):
    imh, imw = img.shape
    if imh == height and imw == width:
        return img
    
    formatted_img = np.full((height, width), 255)
    formatted_img[(height-imh)//2:(height+imh)//2, (width-imw)//2:(width+imw)//2] = img
    return formatted_img
