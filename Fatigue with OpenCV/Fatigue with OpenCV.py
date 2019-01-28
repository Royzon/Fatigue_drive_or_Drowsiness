import cv2
import numpy as np
from scipy import ndimage
import sys
import os
import utils as ut
from PIL import Image, ImageDraw, ImageFont

SKIP_FRAME = 2  #跳帧数量
frame_skip_rate = 0  # 每隔多少帧跳帧
SCALE_FACTOR = 1.5  # 为加速处理，对读到的每一帧按比例缩小

#正脸侧脸检测器
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
sideFace_cascade = cv2.CascadeClassifier("./haarcascade_profileface.xml")
#眼睛检测器
eyes_classifier = cv2.CascadeClassifier("./haarcascade_eye_tree_eyeglasses.xml")

#光照校准
vidsize=(300,450,3)
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

eyeq = []
def push_val(val):
    if (val < 800):
        if len(eyeq) <= 5:
            eyeq.append(val)
        else:
            eyeq.append(val)#添加val到列表末尾
            eyeq.pop(0)#删除第一个元素
    return avg_eyeq()

def avg_eyeq():
    avg = 0
    for i in eyeq:
        avg = avg + i
    avg = avg / len(eyeq)

    return avg

#字典映射用于跟踪旋转图
rotation_maps = {
    "left": np.array([-30, 0, 30]),
    "right": np.array([30, 0, -30]),
    "middle": np.array([0, -30, 30]),
}

#进行角度旋转，并返回优化的旋转图
def get_rotation_map(rotation):
    if rotation > 0: return rotation_maps.get("right", None)
    if rotation < 0: return rotation_maps.get("left", None)
    if rotation == 0: return rotation_maps.get("middle", None)

current_rotation_map = get_rotation_map(0)

webcam = cv2.VideoCapture(0)
# webcam = cv2.VideoCapture("./night2.mp4")
ret, frame = webcam.read() #获取第一帧
frame_scale = (int(frame.shape[1] / SCALE_FACTOR), int(frame.shape[0] / SCALE_FACTOR))  # (y, x)

cropped_face = []
num_of_face_to_collect = 150
num_of_face_saved = 0

#存储路径
profile_folder_path = None

#中文乱码处理
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        #OpenCV图片转换为PIL图片格式
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #使用PIL绘制文字
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        r"C:\Windows\Fonts\simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    #PIL图片格式转换成OpenCV的图片格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

#命令行控制
#  if len(sys.argv) == 1:
#     print("\nError: No Saving Diectory Specified\n")
#     exit()
# elif len(sys.argv) > 2:
#     print("\nError: More Than One Saving Directory Specified\n")
#     exit()
# else:
#     profile_folder_path = ut.create_profile_in_database(sys.argv[1])


while ret:
    # 检测到键盘输入ESC Q q，退出循环
    key = cv2.waitKey(1)
    if key in [27, ord('Q'), ord('q')]:
        break
    #缩小尺寸
    resized_frame = cv2.resize(frame, frame_scale)

    #如果最近一帧没有检测到脸跳帧
    if frame_skip_rate == 0:
        faceFound = False
        for rotation in current_rotation_map:

            frame_rotated = ndimage.rotate(resized_frame, rotation)
            frame_gamma = gamma_correction_auto(frame_rotated, equalizeHist=False)
            frame_gray = cv2.cvtColor(frame_gamma, cv2.COLOR_BGR2GRAY)
            frame_Hist = cv2.equalizeHist(frame_gray)
            frame_filter = cv2.bilateralFilter(frame_Hist, 3, 50, 50)

            # 检测正脸
            faces = face_cascade.detectMultiScale(
                frame_gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),

            )
            #如果没有发现正脸检测侧脸
            faces = faces if len(faces) else sideFace_cascade.detectMultiScale(
                frame_gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces):
                # 对faces二维数组按照人脸面积w*h从小到大排序, 取最大人脸
                faces = faces[(faces[:, 2] * faces[:, 3]).argsort()]
                print("b: ", faces)
                x, y, w, h = faces[-1]
                print("c: ", x, y, w, h)
                # 将边框缩放回原始帧大小
                cropped_face = frame_rotated[y: y + h, x: x + w]
                cropped_face = cv2.resize(cropped_face, (200,200), interpolation=cv2.INTER_AREA)
                cv2.putText(frame_rotated, "Get Face", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
                cv2.rectangle(frame_rotated, (x, y), (x + w, y + h), (0, 255, 0))
                gray = cv2.rectangle(frame_filter, (x, y), (x + w, y + h), (255, 0, 0), 2)
                img = cv2.rectangle(frame_rotated, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # 眼睛位于人脸高度0.2h~0.5h，宽度0.13w~0.87W，对此位置定位以精确识别眼睛
                eh1 = int(h * 0.25)
                eh2 = int(h * 0.5)
                ew1 = int(w * 0.13)
                ew2 = int(w * 0.87)

                # 画出眼睛区域
                '''矩形画出区域 rectangle参数（图像，左顶点坐标(x,y)，右下顶点坐标（x+w,y+h），线条颜色，线条粗细）'''
                cv2.rectangle(img, (x + ew1, y + eh1), (x + ew2, y + eh2), (0, 255, 0), 2)

                '''img获取坐标为，【y,y+h之间（竖）：x,x+w之间(横)范围内的数组】
                               img_facehalf_left 截取左眼区域，即图像的右半区域
                               img_facehalf_right截取右眼区域，在图像的左半区域
                               img_facehalf_bottom是截取人脸识别到区域下半部分'''
                roi_gray = gray[y + eh1:y + eh2, x + ew1:x + ew2]
                roi_color = img[y + eh1:y + eh2, x + ew1:x + ew2]

                # 检测器识别眼睛
                eyes = eyes_classifier.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=1, minSize=(20, 20),
                                                        maxSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE)

                max_eyes = 2
                cnt_eye = 0
                # 检测到眼睛后循环
                for ex, ey, ew, eh in eyes:
                    if (cnt_eye == max_eyes):
                        break
                    image_name = 'Eye_' + str(cnt_eye)
                    print(image_name)

                    ex = int(ex + (ew / 6))
                    ew = int(ew - (ew / 6))
                    ey = int(ey + (eh / 3))
                    eh = int(eh / 3)
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)

                    roi_eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]
                    roi_eye_color = roi_color[ey:ey + eh, ex:ex + ew]
                    # 图像，通道[0]-灰度图，掩膜-无，灰度级，像素范围,直方图x轴是灰度值（一般0~255），返回值y轴就是图像中每一个灰度级对应的像素点的个数。
                    hist = cv2.calcHist([roi_eye_gray],[0],None,[256],[0, 256])
                    histn = []
                    max_val = 0
                    for i in hist:
                        value = int(i[0])
                        histn.append(value)
                        if (value > max_val):
                            max_val = value

                    for index, value in enumerate(histn):
                        histn[index] = ((value * 256) / max_val)

                    # 阈值定义
                    threshold = np.argmax(histn)
                    print(threshold.shape)

                    # threshold = 65

                    # 二值化
                    roi_eye_gray2 = roi_eye_gray.copy()
                    total_white = 0
                    total_black = 0
                    for i in range(0, roi_eye_gray2.shape[0]):
                        for j in range(0, roi_eye_gray2.shape[1]):
                            pixel_value = roi_eye_gray2[i, j]
                            if (pixel_value >= threshold):
                                roi_eye_gray2[i, j] = 255
                                total_white = total_white + 1
                            else:
                                roi_eye_gray2[i, j] = 0
                                total_black = total_black + 1

                    binary = cv2.resize(roi_eye_gray2, None, fx=3, fy=3)
                    cv2.imshow('binary', binary)
                    if image_name == "Eye_0":
                        ag = push_val(total_white)
                        print(image_name, " : ", total_white, " : ", ag)
                    # print("Black ", total_black)
                    # print("White ", total_white)

                    cnt_eye = cnt_eye + 1

                if len(eyes) == 0:
                    ag = push_val(0)

                average = avg_eyeq()
                if average > 30:
                    print("Eye_X: ", average)
                else:
                    print("---------------------", average)
                    cv2.putText(frame_rotated, "Warning !", (120, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (18, 255, 255), 2)
                    # cv2ImgAddText(frame_rotated, "警告", 140, 60, (255, 255, 0), 200)
                    # winsound.Beep(1000, 100)

                #旋转帧至原来位置并修剪黑色填充物
                resized_frame = ut.trim(ut.rotate_image(frame_rotated, rotation * (-1)), frame_scale)
                #重置旋转后的图像
                current_rotation_map = get_rotation_map(rotation)

                faceFound = True

                break

        if faceFound:
            frame_skip_rate = 0
            print("Face Found")
        else:
            frame_skip_rate = SKIP_FRAME
            print("Face Not Found")

    else:
        frame_skip_rate -= 1
        print("Face Not Found")

    cv2.putText(resized_frame, "Press 'ESC/Q/q' to quit.", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    cv2.imshow("Real Time Facial Recognition", resized_frame)

    if len(cropped_face):
        gray = frame_filter[y: y + h, x: x + w]
        cv2.imshow("Cropped Face", gray)

    #获取下一帧
    ret, frame = webcam.read()

webcam.release()
cv2.destroyAllWindows()