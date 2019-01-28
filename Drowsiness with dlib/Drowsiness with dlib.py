#导入工具包
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import operator
import imutils
import time
import dlib
import cv2
import playsound
from threading import Thread

#将landmark特征点坐标值转换为numpy数组形式
def shape_to_np(shape, dtype="int"):
    #初始化一个68行2列的整型0矩阵
    coords = np.zeros((68, 2), dtype=dtype)

    #遍历landmark68个特征点，提取其坐标值
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

#欧氏距离计算
def euclidean_dist(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

#计算眼睛高宽比，以衡量是否闭眼
def eye_aspect_ratio(eye):
    #利用landmark特征点坐标计算眼睛垂直高度方向两点间的欧氏距离
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

#计算张嘴程度，用以衡量是否打哈欠
def mouth_aspect_ratio(mouth):
    mouth_x = euclidean_dist(mouth[0], mouth[6])
    mouth_y1 = euclidean_dist(mouth[1], mouth[7])
    mouth_y2 = euclidean_dist(mouth[3], mouth[5])
    mou = (mouth_y1 + mouth_y2) / (2.0 * mouth_x)
    return mou

#光照校准
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
    # print(correct_param)
    return output

def warnning():
    path='warning.wav'
    playsound.playsound(path)

#定义用于衡量的阀值
EYE_THRESH = 0.23
EYE_CONSEC_FRAMES = 2

MOUTH_THRESH = 0.7
MOUTH_CONSEC_FRAMES =20

#初始化计数器
EYE_COUNTER = 0
MOUTH_COUNTER = 0
ALARM_ON = False

face_list=[[0] * 5 for i in range(10)]
face_np = np.zeros((10, 5), dtype='int32')

#landmark人脸特征点标记模型
marker_path = "./shape_predictor_68_face_landmarks.dat"
#使用dlib自带的frontal_face_detector作为人脸提取器
detector = dlib.get_frontal_face_detector()
#用官方提供的模型构建特征提取器
marker = dlib.shape_predictor(marker_path)

#启动视频
print("Starting video stream thread...")
#本地视频加载
# vs = FileVideoStream('./me-night.mp4').start()
# fileStream = True

#usb摄像头视频
vs = VideoStream(src=0).start()
fileStream = False

vidsize=(300,450,3)

time.sleep(0.2)

#遍历加载的视频
while True:
    # 检测到键盘输入q，退出循环
    if cv2.waitKey(1) in [27, ord('Q'), ord('q')]:
        break

    #从视频文件中抓取帧，进行预处理
    frame = vs.read()
    frame_resized = cv2.resize(frame, (320,300))
    frame_gamma = gamma_correction_auto(frame_resized, equalizeHist=False)
    frame_gray = cv2.cvtColor(frame_gamma, cv2.COLOR_BGR2GRAY)
    frame_Hist=cv2.equalizeHist(frame_gray)
    frame_filter=cv2.bilateralFilter(frame_Hist, 3, 50, 50)

    #从灰度图中检测出人脸
    rects,scores,idx = detector.run(frame_filter, 1)
    print("Dectet {} faces !".format(len(rects)))

    # 遍历所有灰度图中检测出的人脸，利用二维列表，将矩形框及其角点进行排序
    for i,rect in enumerate(rects):
        print("Dectetion_{},rect:{},score:{},face_type:{}".format(i,rects[i],scores[i],idx[i]))

        # face_list[0][0]=i
        # face_list[0][1]=rect.left()
        # face_list[0][2]=rect.top()
        # face_list[0][3]=rect.right()
        # face_list[0][4]=rect.bottom()
        # print("a: ", face_list)
        #
        # face_np=np.array(face_list)
        # faces = face_np[(face_np[:, 3] - face_np[:, 1]).argsort()]
        # print("b: ", faces)
        # j,left,top,right,bottom = faces[-1]
        # print("c: ", j,left,top,right,bottom)

        # 绘制面部边框，left人脸左边距离图片左边界的距离，right人脸右边距离图片左边界的距离，top人脸上边距离图片上边界的距离，bottom人脸下边距离图片上边界的距离

        # 将dlib矩形转换为OpenCV样式的边界框[即（x，y，w，h）]，然后绘制边界框
        # x, y, w, h = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame_resized,(rect.left(),rect.top()), (rect.right(),rect.bottom()),(255, 0, 0), 2)

        # 确定面部区域的面部标志，然后将面部标志（x，y）坐标转换成NumPy阵列
        face_marked = marker(frame_filter,rect)
        face_marknp = face_utils.shape_to_np(face_marked)

        #提取左眼、右眼、鼻子、嘴巴的坐标
        leftEye = face_marknp[42:48]
        rightEye = face_marknp[36:42]
        mouth_outter = face_marknp[48:60]
        mouth_inner = face_marknp[60:68]

        # 然后使用坐标计算双眼的眼高宽比，用平均化双眼的眼高宽比
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # 计算张嘴程度
        mar = mouth_aspect_ratio(mouth_inner)

        # 使用cv2.drawContours填充眼睛轮廓
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth_inner)
        cv2.drawContours(frame_resized, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame_resized, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame_resized, [mouth_outter], -1, (0, 255, 0), 1)
        cv2.drawContours(frame_resized, [mouth_inner], -1, (0, 255, 0), -1)

        # 如果眼睛纵横比低于特定阈值，记为一次闭眼,并计算闭眼的时长
        if ear < EYE_THRESH:
            EYE_COUNTER += 1
            if EYE_COUNTER >= EYE_CONSEC_FRAMES:
                cv2.putText(frame_resized, "Warning !", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (18, 255, 255), 2)
                # 如果警示铃声本来关闭，当次情况下要开启
                # if not ALARM_ON:
                #     ALARM_ON = True
                #     t = Thread(target=warnning)
                #     t.deamon = True
                #     t.start()

        elif mar > MOUTH_THRESH:
            MOUTH_COUNTER += 1

            if (EYE_COUNTER >= EYE_CONSEC_FRAMES) or (MOUTH_COUNTER >= MOUTH_CONSEC_FRAMES):
                cv2.putText(frame_resized, "Warning !", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (18, 255, 255), 2)
                # if not ALARM_ON:
                #     ALARM_ON = True
                #     t = Thread(target=warnning)
                #     t.deamon = True
                #     t.start()

        else:
            EYE_COUNTER = 0
            MOUTH_COUNTER = 0
            ALARM_ON = False

        # 显示闭眼、张嘴程度
        cv2.putText(frame_resized, "EAR: {:.3f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(frame_resized, "MAR: {:.3f}".format(mar), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # 显示框
    cv2.imshow("Frame", frame_resized)
    # 调整窗口大小video为64*64
    # cv2.resizeWindow('video', 540, 480)

#做一些清理
cv2.destroyAllWindows()
vs.stop()