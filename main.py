from PIL.Image import coerce_e
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

def houghCircle():
  star = cv.imread('circle.png')
  gray_img = cv.cvtColor(star,cv.COLOR_BGR2GRAY)
  img = cv.medianBlur(gray_img,7)

  cirles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,200,param1=100,param2=50,minRadius=0,maxRadius=100)
  for circle in cirles[0,:]:
    cv.circle(star,(circle[0],circle[1]),circle[2],(255,255,0),5) # 画圆
    cv.circle(star,(circle[0],circle[1]),5,(255,255,0),-1) # 画圆心
  plt.imshow(star[:,:,::-1])
  plt.show()

def cornerHarris():
  img = cv.imread('chessboard.png')
  gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
  gray = np.float32(gray)

  dst = cv.cornerHarris(gray,2,3,0.04)
  img[dst>0.001*dst.max()] = [0,0,255]

  plt.imshow(img[:,:,::-1])
  plt.show()

def corner_Shit_Tomas():
  img = cv.imread('cctv.png')
  gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

  corners = cv.goodFeaturesToTrack(gray,1000,0.01,10)
  for corner in corners:
    x,y=corner.ravel()
    cv.circle(img,(x,y),2,(0,0,255),-1)

  plt.imshow(img[:,:,::-1])
  plt.show()

def findMan():
  img = cv.imread("wuli.jpg")  
  gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

  face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")  
  eye_cascade = cv.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")  
    
  faces = face_cascade.detectMultiScale(gray,1.6,10,cv.CASCADE_SCALE_IMAGE,(30,30))  

  if len(faces)>0:  
    for faceRect in faces:  
      x,y,w,h = faceRect  
      cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),10)  

      roi_gray = gray[y:y+h,x:x+w]  
      roi_color = img[y:y+h,x:x+w]  

      eyes = eye_cascade.detectMultiScale(roi_gray,1.1,1,cv.CASCADE_SCALE_IMAGE,(2,2))  
      for (ex,ey,ew,eh) in eyes:  
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)  
              
  plt.imshow(img[:,:,::-1])
  plt.show()

def findManInVidel():
  cap = cv.VideoCapture('dasima.mp4')

  face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    frame = cv.resize(frame, None, fx=0.5, fy=0.5)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_zone = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)
    for x, y, w, h in face_zone:
      cv.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=[0, 0, 255], thickness=2)
      cv.circle(frame, center=(x + w // 2, y + h // 2), radius=w // 2, color=[0, 255, 0], thickness=2)
    cv.imshow('video', frame)
    if ord('q') == cv.waitKey(40):
      break

  cv.destroyAllWindows()
  cap.release()


def camera():
  # 识别电脑摄像头并打开
  cap = cv.VideoCapture(1)
  # 创建一个级联分类器 加载一个.xml分类器文件 它既可以是Haar特征也可以是LBP特征的分类器
  face_detect = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

  while True:
      # 读取视频片段
      flag, frame = cap.read()
      frame = cv.flip(frame, 1)
      if not flag:   # 读完视频后falg返回False
          break
      # 灰度处理
      gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
      # 多个尺度空间进行人脸检测   返回检测到的人脸区域坐标信息
      face_zone = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
      # 绘制矩形和圆形检测人脸
      for x, y, w, h in face_zone:
          cv.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=[0, 0, 255], thickness=2)
          cv.circle(frame, center=(x + w // 2, y + h // 2), radius=w // 2, color=[0, 255, 0], thickness=2)
      # 显示图片
      cv.imshow('video', frame)
      # 设置退出键q 展示频率
      if ord('q') == cv.waitKey(30):
          break

  # 释放资源
  cv.destroyAllWindows()
  cap.release()
# readImg()
# draw()
# hist()
# sobel()
# laplacian()
# canny()
# match()
# houghCircle()
# cornerHarris()
# corner_Shit_Tomas()
# findMan()
# findManInVidel()
camera()