import cv2
import os

img_path = os.path.join(os.path.dirname(__file__), 'imgs')
img = cv2.imread(os.path.join(img_path, os.listdir(img_path)[2300]))
mod_img = img.copy()
grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
mser = cv2.MSER().create()
mser.setMaxArea(30)
mser.setMinArea(1)
_, rects = mser.detectRegions(thresh)
for (x,y,w,h) in rects:
    cv2.rectangle(img, (x,y), (x+w,y+h), color=(0,0,0), thickness=1)
    mod_img[y:y+h,x:x+w,:].fill(255)

grey = cv2.cvtColor(mod_img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(grey,(5,5),0)
_, thresh = cv2.threshold(grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
mser = cv2.MSER().create()
mser.setMaxArea(mod_img.shape[0]*mod_img.shape[1]//2)
mser.setMinArea(150)
_, rects = mser.detectRegions(thresh)
for (x,y,w,h) in rects:
    cv2.rectangle(mod_img, (x,y), (x+w,y+h), color=(0,0,0), thickness=1)

cv2.imshow("modimg", mod_img)
cv2.imshow("otsu", thresh)
cv2.waitKey()