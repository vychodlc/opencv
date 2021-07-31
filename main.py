import cv2 as cv
import numpy as np

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

# readImg()
# draw()