## OpenCV 基本操作

### 基础操作

#### 1. 图像IO操作

```python
cv.imread() # 读取图像
cv.imshow() # 显示图像
# 为了使图片常显，可以使用 waitKey
cv.waitKey() # 通常参数设置为0
cv.imwrite() # 保存图像
```

#### 2. 图像绘制操作

```python
cv.line(img, start_pos, end_pos, gbr, line_width) # 画线
cv.circle(img, cneter, radius, gbr, line_width) # 画圆
cv.rectangle(img, left_top_pos, right_bottom_pos, gbr, line_width) # 画矩形
cv.putText(img, text, pos, font, size, gbr, line_width, method) # 添加文字

# line_width 设置为 -1 指填充
```

#### 3. 图像的属性

形状：img.shape
大小：img.size
数据类型：img.dtype

#### 4. 拆分合并（split, merge）

#### 5. 色彩空间的改变

```python
cv.cvtColor(img, flag)
```

### 算术操作

#### 6. 图像的加法（图片相同大小）

```python
x = np.uint8([250])
y = np.uint8([10])
print( cv.add(x,y) ) # [[255]] 
# 250+10=260 => 255 有上限
print( x+y ) # [4] 
# 250+10=260 => 260%256=4

# cv.add(x+y) 是 OpenCV 方法，是饱和操作
# x+y 是 Numpy 方法，是模运算
```

#### 7. 图像混合（图片相同大小）

`混合` 也可以理解为是 `加法` ，但是不同的地方在于作为加数的二者在混合操作中可以有不同的权重
$$
g(x) = (1-\alpha)f_0(x)+\alpha f_1(x)
$$
通过修改 $\alpha$ 的值就可以修改二者的权重

```python
cv.addWeighted(img1, 0.7, img2, 0.3, 0) # 最后的一个 0 的是 γ 的值
```

## OpenCV 图像处理

### 几何变换

#### 1. 图像缩放

```python
cv2.resize(src, dsize, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)

# src：输入图像
# dsize：绝对尺寸，直接指定调整后图像的大小
# fx,fy：相对尺寸，需要将 dsize 设置为 None，然后将 fx，fy 设置成比例因子
# interpolation：插值方法
	# cv2.INTER_LINEAR		双线性插值
    # CV2.INTER_NEAREST		最近邻插值
    # CV2.INTER_AREA		像素区域重采样（默认）
    # CV2.INTER_CUBIC		双三次插值
    
# 示例
rows,cols = img1.shape[:2]
cv2.resize(img1, (2*cols,2*rows), interpolation=cv.INTER_CUBIC)

cv2.resize(img1, None, fx=0.5, fy=0.5)
```

#### 2. 图像平移

```python
cv2.warpAffine(img, M, dsize)

# img 输入图像
# M 2*3 移动矩阵
# dsize 输出图像的大小

# 示例
M = np.float32([[1,0,100],[0,1,50]])  # 图像移动（50，100）的距离
```

#### 3. 图像旋转

$$
x'=r\cos(\alpha-\theta)\\
y'=r\sin(\alpha-\theta)\\
\\
r = \sqrt{x^2+y^2}, \sin\alpha=\frac{y}{\sqrt{x^2+y^2}},\cos\alpha=\frac{x}{\sqrt{x^2+y^2}}\\
\\
x'=x\cos\theta+y\sin\theta\\
y'=-x\sin\theta+y\cos\theta\\
$$

```python
# 得到旋转矩阵
M = cv2.getRotationMatrix20(center, angle, scale) # 旋转中心，旋转角度，缩放比例
# 使用矩阵进行变换
cv2.warpAdffine(img, M, (cols,rows))
```

>  原点位置在左上角

#### 4. 仿射变换

`仿射变换` 主要是对图像的缩放，旋转，翻转，平移等操作的一个组合。

> 变换矩阵是 3*2 的矩阵

$$
M =
\begin{bmatrix}
A & B
\end{bmatrix}
=
\begin{bmatrix}
a_{00} & a_{01} & b_0 \\
a_{10} & a_{11} & b_1 \\
\end{bmatrix}
$$

这其中的 $A$ 是线性变换矩阵，$B$ 是平移项

>  原点位置在右上角

```python
pos1 = np.float32([[50,50],[200,50],[50,200]])		# 三个点
pos2 = np.float32([[100,100],[200,50],[100,250]])	# 三个点
# 通过两个点获取变换矩阵
M = cv2.getAffineTransform(pos1, pos2)
# 使用变换矩阵对整个图形进行仿射变换
cv.warpAffine(img, M, (cols,rows))
```

#### 5. 透射变换

`透射变换` 是视角变化的结果，可以理解为将图像投影到一个新的视平面

> 变换矩阵是 3*3 的矩阵

```python
pos1 = ... # 四个点
pos2 = ... # 四个点
# 通过两个点获取透射变换矩阵
T = cv2.getPerspectiveTransform(pos1,pos2)
# 使用变换矩阵对原图形进行透射变换
cv2.warpAffine(img, T, (cols,rows))
```

#### 6. 图像金字塔

`图像金字塔` 主要用于图像的分割，是一种以多分辨率来解释图像的有效但概念简单的结构

金字塔的底部是待处理图像的高分辨率表示，而顶部是低分辨率的近似，层级越高，图像越小，分辨率越低

```python
cv2.pyrUp(img)		# 对图像进行上采样 图像变大
cv2.pyrDown(img)	# 对图像进行下采样 图像变小
```

### 形态学操作

+ 理解图像的邻域、连通性
+ 了解不同的形态学操作：腐蚀、膨胀、开闭运算、礼帽黑帽

#### 1. 连通性

+ `4邻域`：上下左右

+ `D邻域`：左上、左下、右上、右下

+ `8邻域`：九宫格的其他八个点

像素连通的必要条件

+ 像素位置相邻
+ 像素的灰度值满足特定的 `似性准则` （通常是灰度值相同）

连通种类：

+ `4连通`
+ `8连通`
+ `m连通`（4连通 和 D连通 的混合连通）
  + q 在 p 的 4邻域 中
  + 或
  + q 在 p 的 D邻域中，但是二者的 4邻域 交集为空

#### 2. 形态学操作

##### 腐蚀和膨胀

```python
cv2.erode(img, kernel, iteration)	# 腐蚀的作用：消除物体边界点，使目标缩小。可以消除小于结构元素的噪声点
cv2.dilate(img, kernel, iteration)	# 膨胀的作用：将与物体接触的所有背景点合并到物体中，使目标增大，可添补目标中的孔洞
```

##### 开闭运算

```python
# 开运算：先腐蚀后膨胀；
# 作用：分离物体、消除小区域；
# 特点：消除噪点，去除小的干扰块，而不影响原图像
cv2.morphologyEx(img, cv.MORPH_OPEN, kernel)
# 闭运算：先膨胀后腐蚀；
# 作用：消除闭合物体中的孔洞；
# 特点：可以填充闭合区域
cv2.morphologyEx(img, cv.MORPH_CLOSE, kernel)
```

##### 礼帽和黑帽

```python
# 礼帽 = 原图 - 开运算
# 突出了比原图轮廓周围的区域更明亮的区域
# 用来分离比邻近点亮一些的斑块（背景提取）
cv2.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
# 黑帽 = 闭运算 - 原图
# 突出了比原图轮廓周围的区域更暗的区域
# 用来分离比邻近点暗一些的斑块
cv2.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
```

#### 3. 图像平滑

##### 图像噪声

+ 椒盐噪声：图像中随机出现的白点或者黑点
+ 高斯噪声：噪声的概率密度分布使正态分布

##### 图像平滑

从信号处理的角度看就是去除其中的高频，保留低频；即可以使用低通滤波器

+ 均值滤波：去噪同时去除了很多细节部分，图像会变模糊

  ```python
    cv.blur(src, ksize, anchor, borderType)
    # 输入图像，卷积核的大小（(x,y)），核中心（默认值(-1，-1)），填充边界类型
  ```

+ 高斯滤波：去除高斯噪声

  ```python
    # 首先确定权重矩阵，还需要进行归一化处理，使9个权值求和为1
    # 对某一中心点及其周围区域乘上权重矩阵即可得到中心点的新值
  
    cv.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType)
    # 输入图像，卷积核的大小（(x,y)(宽度和高度都应为奇数)），水平方向的标准差，垂直方向的标准差（0的话就表示与sigmaX相等），填充边界类型
  ```

+ 中值滤波：去除椒盐噪声

  ```python
    # 非线性滤波，用像素点邻域灰度值的中值来代替该像素点的灰度值
    # 中值滤波对椒盐噪声尤其有用，因为它不依赖于邻域内那些与典型值差别很大的值
      
    cv.medianBlur(src, ksize)
    # 输入图像，卷积核的大小（(x,y)）
  ```

+ 双边滤波


#### 4. 直方图

##### 灰度直方图

横坐标为灰度值

> dims：需要统计的特征数目
> bins：每个特征空间子区段的数目，可理解为“直条”或“组距”
> range：要统计特征的取值范围

```python
cv2.calcHist(images, channels, mask, histSize, ranges[,hist[,accumulate]])

# 示例
cv2.calcHist([img], [0], None, [256], [0,256])
```

**掩膜的应用**

`掩膜` 是用选定的图像、图形或物体，对要处理的图像进行遮挡，来控制图像处理的区域

主要用途：提取感兴趣区域、屏蔽作用、结构特征提取、特殊形状图像制作

步骤：1、以灰度图读取图像；2、创建蒙版（设置为0、1）；3、进行掩膜；4、统计掩膜后的灰度直方图

##### 直方图均衡化

对图像进行非线性拉伸，重新分配图像像素值，使一定灰度范围内的像素数量大致相同

作用：提高图像整体的对比度；可以在X光图像中提高骨架结构的显示；另外在曝光不足或过度的图像中可以更好的突出细节

```python
cv2.qualizeHist(img)
```

**自适应的直方图均衡化**

将图像分割成很多小块，对每个小块分别进行直方图均衡化

```python
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 
# clipLimit: 对比度限制
# tileGridSize: 分块的大小
cl = clahe.apply(img)
```

#### 5. 边缘检测

目的：标识数字图像中亮度明显变化的点

图像边缘检测大幅度地减少了数据量，并且剔除了可以认为不相关的信息，保留了图像重要的结构属性。主要分为两类：基于搜索、基于零穿越

+ 基于搜索：通过寻找图像一阶导数中的最大值来检测边界，然后利用计算结果估计边缘的局部方向，通常采用梯度的方向，并利用此方向找到局部梯度模的最大值；代表算法：Sobel算子、Scharr算子
+ 基于零穿越：通过寻找图像二阶导数零穿越来寻找边界；代表算法：Laplacian算子

**Sobel算子**

```python
Sobel_x_or_y = cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta, borderType)

# dx,dy: 0表示这个方向上没有求导，取值为0,1
# ksize 为算子大小，即卷积核的大小，必须为奇数1，3，5，7，9，默认为三；设置为 -1 即使用3*3的 Scharr 算子
# scale 缩放导数的比例常数

# 示例
img = cv.imread('image.jpg',0)
x = cv.Sobel(img, cv.CV_16S, 1, 0) # 由于求导后会有负值，也有可能会有大于255的值，所以数据类型需要设置成 16位有符号 的数据类型，即 cv2.CV_16S
y = cv.Sobel(img, cv.CV_16S, 0, 1)
Scale_absX = cv.convertScaleAbs(x) # 得到 uint8 数据
Scale_absY = cv.convertScaleAbs(y)
result = cv.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
```

**Laplacian算子**

```python
Laplacian(src, dst, ddepth, ksize=1, scale=1, delta=0, borderType)

# 示例
img = cv.imread('image.jpg',0)
result = cv.Laplacian(img, cv.CV_16S)
Scale_abs = cv.convertScaleAbs(result)
```

**Canny边缘检测**

第一步：噪声去除——使用5*5高斯滤波器去除噪声
第二步：计算图像梯度——使用Sobel算子获得x,y两个方向上的一阶导数，从而得到边界的梯度和方向
$$
EdgeGradient(G)=\sqrt{G_x^2+G_y^2}\\
Angle(\theta)=tan^{-1}(\frac{G_y}{G_x})
$$
第三步：非极大值抑制——去除非边界上的点；最终得到具有“细边”的二进制图像
第四步：滞后阈值——设置两个阈值 maxVal,minVal ；高于 maxVal 边界作为真的边界被保留，低于 minVal 的边界抛弃；若介于二者之间，需要判断是否与某个已知的真的边界点相连，若相连则保留作为真的边界，否则抛弃

```python
cv2.Canny(img, threshold1, threshold2) # 需要是灰度图
```

#### 6. 模板匹配与霍夫变换

##### 模板匹配

是在给定的图片中查找和模板最相似的区域——利用滑窗思想，最终将匹配度最高的区域选择为最后的结果

```python
cv2.matchTemplate(img, template, method)

# method
# CV_TM_SQDIFF: 平方差匹配
# CV_TM_CCORR: 相关匹配
# CV_TM_CCOEFF: 利用相关系数匹配
```

##### 霍夫变换

常用于提取图像中的直线和圆等几何形状

笛卡尔坐标系中的一条直线，对应于霍夫空间中的一个点

笛卡尔坐标系中的AB两个点，对应着霍夫空间中的两条直线，这两条直线的交点，这个交点的值（q,b）即代表AB两点所连直线的斜率和截距

笛卡尔坐标系中的ABC三点，对应着霍夫空间中的三条直线，若这三条直线有一公共点，则证明ABC三点共线

为了解决斜率为无穷大的情况，将不使用笛卡尔坐标系而选择使用 `极坐标系` ，这个时候所对应的霍夫空间的是（ρ,θ）的空间
$$
xcos\theta+ysin\theta=\rho
$$
