import cv2

capture = cv2.VideoCapture(0) # 打开摄像头
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # 导入人脸模型
cv2.namedWindow('摄像头') # 获取摄像头画面
while True:
  ret, frame = capture.read() # 读取视频图片
  gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # 灰度
  faces = face.detectMultiScale(gray,1.1,3,0,(100,100))
  for (x, y, w, h) in faces: # 5个参数，一个参数图片 ，2 坐标原点，3 识别大小，4，颜色5，线宽
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), -1)
    cv2.imshow('摄像头', frame) # 显示

    if cv2.waitKey(5) & 0xFF == ord('q'):
      # break
      capture.release() # 释放资源
      cv2.destroyAllWindows() # 关闭窗口
      break