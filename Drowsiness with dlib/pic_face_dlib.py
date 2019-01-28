#导入相关模块
import cv2
import dlib
import numpy as np
import imutils

#直方图均衡化
def imgHist(img):
    # 计算累积直方图
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()

    # 除去直方图中的0值
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = ((cdf_m - cdf_m.min()) / (cdf_m.max() - cdf_m.min())) * 255
    # 将掩模处理掉的元素补为0
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    # 计算
    result = cv2.LUT(img, cdf)
    return result

#光照矫正
def gamma_correction_auto(RGBimage, equalizeHist=False):  # 0.35
    originalFile = RGBimage.copy()
    red = RGBimage[:, :, 2]
    green = RGBimage[:, :, 1]
    blue = RGBimage[:, :, 0]

    forLuminance = cv2.cvtColor(originalFile, cv2.COLOR_BGR2YUV)
    Y = forLuminance[:, :, 0]
    totalPix = vidsize[0] * vidsize[1]
    summ = np.sum(Y[:, :])
    Yaverage = np.divide(totalPix, summ)
    # Yclipped = np.clip(Yaverage,0,1)
    epsilon = 1.19209e-007
    correct_param = np.divide(-0.3, np.log10([Yaverage + epsilon]))
    correct_param = 0.7 - correct_param

    red = red / 255.0
    red = cv2.pow(red, correct_param)
    red = np.uint8(red * 255)
    if equalizeHist:
        red = cv2.equalizeHist(red)

    green = green / 255.0
    green = cv2.pow(green, correct_param)
    green = np.uint8(green * 255)
    if equalizeHist:
        green = cv2.equalizeHist(green)

    blue = blue / 255.0
    blue = cv2.pow(blue, correct_param)
    blue = np.uint8(blue * 255)
    if equalizeHist:
        blue = cv2.equalizeHist(blue)

    output = cv2.merge((blue, green, red))
    print(correct_param)
    return output


def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)

    # Retrieve the number of color channels of the image.
    # channel_count = img.shape[2]

    # color used to fill polygon
    match_mask_color = 255

    # Fill the polygon with white
    cv2.fillPoly(mask, vertices, (255, 255, 255))

    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

PREDICTOR_PATH = ".\shape_predictor_68_face_landmarks.dat"
# 使用dlib自带的frontal_face_detector作为人脸提取器
detector = dlib.get_frontal_face_detector()
# 使用官方提供的模型构建特征提取器
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# 加载图片,重置大小, 转换为灰度图并均衡化
img = cv2.imread(".\me-night.jpg")
# img = imutils.resize(img, width=300,height=500)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_Hist=imgHist(img)
vidsize=(300,450,3)

#defining corners for ROI
height = img.shape[1]
width = img.shape[0]

topLeftPt = (0, height*0.4)
topRightPt = (width, height*0.4)

region_of_interest_points = [
(0, 1.25*height),
topLeftPt,
topRightPt,
(width, 1.25*height),
]

img= gamma_correction_auto(img,equalizeHist = False)
img= region_of_interest(img, np.array([region_of_interest_points], np.int32))
img= cv2.bilateralFilter(img, 9, 80, 80)

# 使用detector进行人脸检测 rects为返回的结果
rects = detector(img, 1)

# 输出人脸数，dets的元素个数即为脸的个数
class NoFaces(Exception):
    pass

if len(rects) == 0:
    raise NoFaces

if len(rects) >= 1:
    print("{} faces detected".format(len(rects)))

for rect in rects:

    # 使用predictor进行人脸关键点识别
    landmarks = np.matrix([[p.x,p.y] for p in predictor(img,rect).parts()])
    img = img.copy()
    # 将dlib形式的方框转换为opencv形式的方框，用以绘制面部边框
    cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 255, 0), 1)

    # 使用enumerate 函数遍历序列中的元素，并提取它们的坐标，以及显示下标
    for idx, point in enumerate(landmarks, 1):
        pos = (point[0, 0], point[0, 1])
        print('第{0}个点的坐标是{1}'.format(idx, pos))
        cv2.putText(img, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.95, color=(255, 0, 0))
        # 绘制特征点
        cv2.circle(img, pos, 15, color=(0, 0, 255))

# 显示图片
cv2.imwrite("img2.jpg", img)
cv2.waitKey(0)