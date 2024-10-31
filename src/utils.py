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
    img = cv2.GaussianBlur(img,(3,3),0)
    _, img = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY)

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
    #chosen based on trial and error
    bounds = {
        'NOISE':10,
        'DOUBLE':75,
        'TRIPLE':105,
        'QUADRUPLE':135,
        'QUINTUPLE':165
    }
    chars = []

    mser = cv2.MSER().create()
    mser.setMaxArea(img.shape[0]*img.shape[1]//4)
    mser.setMinArea(150)
    _, rects = mser.detectRegions(img)
    rects = sorted(rects, key = lambda x: x[0])
    for x, y, width, height in rects:
        if width <= bounds['NOISE'] or height <= bounds['NOISE']:
            continue
        elif width < bounds['DOUBLE']:
            chars.append(img[y:y+height, x:x+width])
        elif bounds['DOUBLE'] <= width < bounds['TRIPLE']:
            n = 2
            sep = width//n + 5
            start = x
            for _ in range(n):
                chars.append(img[y:y+height, start:start+sep])
                start += sep
        elif bounds['TRIPLE'] <= width < bounds['QUADRUPLE']:
            n = 3
            sep = width//n + 5
            start = x
            for _ in range(n):
                chars.append(img[y:y+height, start:start+sep])
                start += sep
        elif bounds['QUADRUPLE'] <= width < bounds['QUINTUPLE']:
            n = 4
            sep = width//n + 5
            start = x
            for _ in range(n):
                chars.append(img[y:y+height, start:start+sep])
                start += sep
        elif width >= bounds['QUINTUPLE']:
            n = 5
            sep = width//n + 5
            start = x
            for _ in range(n):
                chars.append(img[y:y+height, start:start+sep])
                start += sep

    return chars

def center_pad(img, width=96, height=96):
    imh, imw = img.shape
    if imh == height and imw == width:
        return img
    
    formatted_img = np.full((height, width), 255)
    formatted_img[(height-imh)//2:(height+imh)//2, :imw] = img
    return formatted_img
