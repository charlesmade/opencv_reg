import cv2
import matplotlib.pyplot as plt
import imutils
from imutils.perspective import four_point_transform
import random

def img_show(img):
    cv2.namedWindow("MyPic")
    cv2.imshow('MyPic', img)
    # 函数waitKey是等待用户按键触发，函数参数是等待时间，单位时间为ms
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit(0)

def read_img(img_src):
    img = cv2.imread(img_src)
    if img is None:
        capture_img = cv2.VideoCapture(img_src)
        if capture_img.isOpened():
            ret, img = capture_img.read()
            if img is None:
                return False
            else:
                return img
        else:
            return False
    else:
        return img

img_src = 'http://photo.yousi.com/2018-11-09_5be53e39b6ed7.jpg'
#读入图片
image = read_img(img_src)
# print(image.shape) # 图像的宽度、高度和通道数
# print(image.size) # 图像的像素大小
# print(image.dtype) # 图像的数据类型

# print(image)
# exit(0)
#转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#高斯滤波
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

#自适应二值化方法
blurred=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,19,2)

'''
adaptiveThreshold函数：第一个参数src指原图像，原图像应该是灰度图。
    第二个参数x指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
    第三个参数adaptive_method 指： CV_ADAPTIVE_THRESH_MEAN_C 或 CV_ADAPTIVE_THRESH_GAUSSIAN_C
    第四个参数threshold_type  指取阈值类型：必须是下者之一  
                                 •  CV_THRESH_BINARY,
                        • CV_THRESH_BINARY_INV
     第五个参数 block_size 指用来计算阈值的象素邻域大小: 3, 5, 7, ...
    第六个参数param1    指与方法有关的参数。对方法CV_ADAPTIVE_THRESH_MEAN_C 和 CV_ADAPTIVE_THRESH_GAUSSIAN_C， 它是一个从均值或加权均值提取的常数, 尽管它可以是负数。
'''
#这一步可有可无，主要是增加一圈白框，以免刚好卷子边框压线后期边缘检测无果。好的样本图就不用考虑这种问题
blurred=cv2.copyMakeBorder(blurred,5,5,5,5,cv2.BORDER_CONSTANT,value=(255,255,255))

#canny边缘检测
edged = cv2.Canny(blurred, 10, 100)
# img_show(edged)
# 从边缘图中寻找轮廓，然后初始化答题卡对应的轮廓
'''
findContours
image -- 要查找轮廓的原图像
mode -- 轮廓的检索模式，它有四种模式：
     cv2.RETR_EXTERNAL  表示只检测外轮廓                                  
     cv2.RETR_LIST 检测的轮廓不建立等级关系
     cv2.RETR_CCOMP 建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，
              这个物体的边界也在顶层。
     cv2.RETR_TREE 建立一个等级树结构的轮廓。
method --  轮廓的近似办法：
     cv2.CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max （abs (x1 - x2), abs(y2 - y1) == 1
     cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需
                       4个点来保存轮廓信息
      cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
'''
cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cnts = cnts[0] if imutils.is_cv2() else cnts[1]
docCnt = None
# 确保至少有一个轮廓被找到
if len(cnts) > 0:
    # 将轮廓按大小降序排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # 对排序后的轮廓循环处理
    for c in cnts:
        # 获取近似的轮廓
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # 如果近似轮廓有四个顶点，那么就认为找到了答题卡
        if len(approx) == 4:
            docCnt = approx
            break

newimage=image.copy()
for i in docCnt:
    #circle函数为在图像上作图，新建了一个图像用来演示四角选取
    cv2.circle(newimage, (i[0][0],i[0][1]), 50, (255, 0, 0), -1)

paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))
const_gray = 99
# 对灰度图应用二值化算法
thresh=cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,const_gray,2)

#重塑可能用到的图像
width1 = 2400
height1 = 2800
thresh = cv2.resize(thresh, (width1, height1), cv2.INTER_LANCZOS4)
paper = cv2.resize(paper, (width1, height1), cv2.INTER_LANCZOS4)
warped = cv2.resize(warped, (width1, height1), cv2.INTER_LANCZOS4)
# cv2.namedWindow("Image")
# cv2.imshow("Image", thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# exit(0)

#均值滤波
ChQImg = cv2.blur(thresh, (23, 23))

#二进制二值化
ChQImg = cv2.threshold(ChQImg, 100, 225, cv2.THRESH_BINARY)[1]

'''
    threshold参数说明
    第一个参数 src    指原图像，原图像应该是灰度图。
   第二个参数 x      指用来对像素值进行分类的阈值。
   第三个参数 y      指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
   第四个参数 Methods  指，不同的不同的阈值方法，这些方法包括：
                •cv2.THRESH_BINARY        
                •cv2.THRESH_BINARY_INV    
                •cv2.THRESH_TRUNC        
                •cv2.THRESH_TOZERO        
                •cv2.THRESH_TOZERO_INV    
'''
# key = 2
# cv2.imwrite('./img/test2-1-'+str(key)+'.jpg', ChQImg)
# img_show(ChQImg)
Answer = []
# 在二值图像中查找轮廓
cnts = cv2.findContours(ChQImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnts = cnts[0] if imutils.is_cv2() else cnts[1]

for c in cnts:
     # 计算轮廓的边界框，然后利用边界框数据计算宽高比
      (x, y, w, h) = cv2.boundingRect(c)
      # print(x,y,w,h)
      if (w > 60 & h > 20)and y>105 and y<2333 and ((x > 95 and x < 367) or (x > 1370 and x < 1642)):
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            #绘制中心及其轮廓
            cv2.drawContours(paper, c, -1, (0, 0, 255), 5, lineType=0)
            cv2.circle(paper, (cX, cY), 7, (255, 255, 255), -1)
            #保存题目坐标信息
            Answer.append((cX, cY))
x_ray = []
y_ray = []
for ans in Answer:
    x_ray.append(ans[0])
    y_ray.append(ans[1])

plt.scatter(x_ray, y_ray, marker = 'o', color = 'green')
for a, b in zip(x_ray, y_ray):
    plt.text(a, b, (a,b),ha='center', va='bottom', fontsize=10)
plt.show()
exit(0)

def judgex(x, x1, x2):
    if abs((x - x1)) < abs(x - x2):
        return 1
    else:
        return 2

def judgey(y, y1, y2):
    if abs((y - y1)) < abs(y - y2):
        return 4
    else:
        return 8


IDAnswer=[]
xtt1 = [95, 367, 1370, 1642]
ytt1 = [(145 * i) + 105 for i in range(0, 16)]
# xt1 = [120 * i for i in range(0, 21)]
# yt1 = [(73.33 * i) + 900 for i in range(0, 16)]

for i in Answer:
    print(i)
    for j in range(0,len(xtt1)-1):
        if i[0]>xtt1[j] and i[0]<xtt1[j+1]:
            for k in range(0,len(ytt1)-1):
                if i[1]>ytt1[k] and i[1]<ytt1[k+1]:
                    IDAnswer.append((k,j,(judgex(i[0], xtt1[j], xtt1[j+1]) + judgey(i[1], ytt1[k], ytt1[k+1]))))
#对答案部分重新排序，以最好的方式输出

IDAnswer.sort()
print(IDAnswer)
print(len(IDAnswer))