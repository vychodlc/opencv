import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def readImg():
  im = cv.imread('image.jpg',0)
  cv.imshow('image',im)
  k = cv.waitKey(0)
  if k ==27:
    cv.destroyAllWindows()

def draw():
  img = np.zeros((512,512,3),np.uint8)

  cv.line(img,(0,0),(512,100),(255,0,0),5)
  cv.circle(img,(255,255),100,(255,255,0),-1)
  cv.rectangle(img,(0,255),(100,512),(255,0,255),1)
  cv.putText(img,'hello',(100,150),cv.FONT_HERSHEY_COMPLEX,5,(0,0,255),3)

  cv.imshow('image',img)
  k = cv.waitKey(0)
  if k ==27:
    cv.destroyAllWindows()

def hist():
  img = cv.imread('image.jpg',0)
  histr = cv.calcHist([img],[0],None,[256],[0,256])
  plt.figure(figsize=(10,6),dpi=100)
  plt.plot(histr)
  plt.grid()
  plt.show()

def sobel():
  img = cv.imread('image.jpg',0)
  x = cv.Sobel(img, cv.CV_16S, 1, 0)
  y = cv.Sobel(img, cv.CV_16S, 0, 1)
  Scale_absX = cv.convertScaleAbs(x)
  Scale_absY = cv.convertScaleAbs(y)
  result = cv.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
  plt.imshow(result,cmap=plt.cm.gray)
  plt.show()
  
def laplacian():
  img = cv.imread('image.jpg',0)
  result = cv.Laplacian(img, cv.CV_16S)
  Scale_abs = cv.convertScaleAbs(result)
  plt.imshow(Scale_abs,cmap=plt.cm.gray)
  plt.show()

def canny():
  img = cv.imread('image.jpg',0)
  canny = cv.Canny(img, 0, 100)
  plt.imshow(canny,cmap=plt.cm.gray)
  plt.show()

def match():
  img = cv.imread('image2.jpg')
  template = cv.imread('template3.jpg')
  h,w = template.shape[:2]

  res = cv.matchTemplate(img, template, cv.TM_CCORR)
  min_val,max_val,min_loc,max_loc = cv.minMaxLoc(res)

  top_left = max_loc
  bottom_right = (top_left[0]+w,top_left[1]+h)

  cv.rectangle(img,top_left,bottom_right,(0,255,0),2)

  print(top_left,bottom_right)

  plt.imshow(img[:,:,::-1])
  plt.show()


# readImg()
# draw()
# hist()
# sobel()
# laplacian()
# canny()
match()