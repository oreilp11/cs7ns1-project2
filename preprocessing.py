import cv2
import os

IMG_PATH = os.path.join(os.path.dirname(__file__), 'imgs')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(OUT_PATH):
    os.mkdir(OUT_PATH)


img = cv2.imread(os.path.join(IMG_PATH, os.listdir(IMG_PATH)[1]))
mod_img = img.copy()
grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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
_, thresh = cv2.threshold(grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#next applying MSER on large areas to select characters
mser = cv2.MSER().create()
mser.setMaxArea(mod_img.shape[0]*mod_img.shape[1]//2)
mser.setMinArea(150)
_, rects = mser.detectRegions(thresh)
for (x,y,w,h) in rects:
    cv2.rectangle(mod_img, (x,y), (x+w,y+h), color=(0,0,0), thickness=1)
#can crop by bounding box/padded bounding box now

cv2.imwrite(os.path.join(OUT_PATH,'original.png'),mod_img)
cv2.imwrite(os.path.join(OUT_PATH,'thresh.png'),thresh)

#displaying images
#cv2.imshow("modimg", mod_img)
#cv2.imshow("otsu", thresh)
#cv2.waitKey()