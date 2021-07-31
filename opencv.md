## OpenCV 基本操作



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

#### 6. 图像的加法

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

#### 7. 图像混合

`混合` 也可以理解为是 `加法` ，但是不同的地方在于作为加数的二者在混合操作中可以有不同的权重
$$
g(x) = (1-\alpha)f_0(x)+\alpha f_1(x)
$$
通过修改 $\alpha$ 的值就可以修改二者的权重

```python
cv.addWeighted(img1, 0.7, img2, 0.3, 0) # 最后的一个 0 的是 γ 的值
```

