import cv2
import numpy as np
from scipy import ndimage
import sys
import os
import utils as ut

webcam = cv2.VideoCapture(0)# 获取实时摄像
# video='with_glass.mp4'#获取本地视频1
# webcam = cv2.VideoCapture(video)#获取本地视频2
ret, frame = webcam.read() #获取第一帧

SCALE_FACTOR = 1.5  # 为加速处理，对读到的每一帧按比例缩小
frame_scale = (int(frame.shape[1] / SCALE_FACTOR), int(frame.shape[0] / SCALE_FACTOR))  # (y, x)
cropped_face = []
num = 0
num_of_save = 150
SKIP_FRAME = 2  #跳帧数量
frame_skip_rate = 0  # 每隔多少帧跳帧

#folder_path = None#存储路径
folder_path_face=r"E:\HASCO\project\code\Fatigue\2_OpenCV get face features\pics\faces"
folder_path_eye=r"E:\HASCO\project\code\Fatigue\2_OpenCV get face features\pics\eyes"
folder_path_mouth=r"E:\HASCO\project\code\Fatigue\2_OpenCV get face features\pics\mouths"

#命令行控制
#  if len(sys.argv) == 1:
#     print("\nError: No Saving Diectory Specified\n")
#     exit()
# elif len(sys.argv) > 2:
#     print("\nError: More Than One Saving Directory Specified\n")
#     exit()
# else:
#     folder_path = ut.create_profile_in_database(sys.argv[1])

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

#获取摄像头返回的宽和高,确定保存视频的格式
# ''' cv.VideoWriter参数（视频存放路径，视频存放格式，fps帧率，视频宽高）
#     注意点1：OpenCV只支持avi的格式，而且生成的视频文件不能大于2GB，而且不能添加音频
#     注意点2：若填写的文件名已存在，则该视频不会录制成功，但可正常使用'''
# size = (int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5), int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video = cv2.VideoWriter(r"E:\HASCO\project\code\Fatigue\2_OpenCV get face features\aaa.avi", fourcc, 5, size)

#正脸和侧脸识别
face_classifier = cv2.CascadeClassifier(r"C:\Program Files\opencv\opencv-master\data\haarcascades\haarcascade_frontalface_alt.xml")
sideFace_classifier = cv2.CascadeClassifier(r"C:\Program Files\opencv\opencv-master\data\haarcascades\haarcascade_profileface.xml")
#左眼和右眼识别
lefteye_classifier = cv2.CascadeClassifier(r"C:\Program Files\opencv\opencv-master\data\haarcascades\haarcascade_lefteye_2splits.xml")
righteye_classifier = cv2.CascadeClassifier(r"C:\Program Files\opencv\opencv-master\data\haarcascades\haarcascade_righteye_2splits.xml")
#嘴巴识别
mouth_classifier=cv2.CascadeClassifier(r"C:\Program Files\opencv\opencv_contrib-master\modules\face\data\cascades\haarcascade_mcs_mouth.xml")

while ret:
    # 检测到键盘输入ESC Q q，退出循环
    key = cv2.waitKey(1)
    if key in [27, ord('Q'), ord('q')]:
        break
    #缩小尺寸
    frame_resized = cv2.resize(frame, frame_scale)

    #如果最近一帧没有检测到脸跳帧
    if frame_skip_rate == 0:
        faceFound = False
        for rotation in current_rotation_map:

            frame_rotated = ndimage.rotate(frame_resized, rotation)
            frame_gamma = gamma_correction_auto(frame_rotated, equalizeHist=False)
            frame_gray = cv2.cvtColor(frame_gamma, cv2.COLOR_BGR2GRAY)
            frame_Hist = cv2.equalizeHist(frame_gray)
            frame_filter = cv2.bilateralFilter(frame_Hist, 3, 50, 50)

            # 检测正脸
            faces = face_classifier.detectMultiScale(
                frame_filter,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
            )

            #如果没有发现正脸检测侧脸
            faces = faces if len(faces) else sideFace_classifier.detectMultiScale(
                frame_filter,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces):
                # 按照人脸面积w*h从小到大排序,取最大人脸
                faces = faces[(faces[:, 2] * faces[:, 3]).argsort()]
                print("b: ", faces)
                x, y, w, h = faces[-1]
                print("c: ", x, y, w, h)
                # 将边框缩放回原始帧大小
                cropped_face = frame_rotated[y: y + h, x: x + w]
                cropped_face = cv2.resize(cropped_face, (200,200), interpolation=cv2.INTER_AREA)

                # 显示并保存捕捉到的人脸
                cv2.imshow("Face", cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY))
                face_to_save = cv2.resize(cropped_face, (50, 50), interpolation=cv2.INTER_AREA)
                # cv2.imwrite('%s/%d.jpg' % (folder_path_face, num),face_to_save)

                cv2.rectangle(frame_rotated, (x, y), (x + w, y + h), (0, 255, 0))
                face_gray = cv2.rectangle(frame_filter, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face_color = cv2.rectangle(frame_rotated, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # 眼睛位于人脸高度0.2h~0.5h，宽度0.13w~0.87W，对此位置定位以精确识别眼睛
                eh1 = int(h * 0.25)
                eh2 = int(h * 0.5)
                ew1 = int(w * 0.13)
                ew2 = int(w * 0.5)
                ew3 = int(w * 0.87)
                # 同样的，截取人脸区域下半部分，以精确识别嘴巴的位置
                mh1 = int(h * 0.7)
                mh2 = int(h * 1.0)
                mw1 = int(w * 0.25)
                mw2 = int(w * 0.75)

                '''img获取坐标为，【y,y+h之间（竖）：x,x+w之间(横)范围内的数组】
                               img_facehalf_left 截取左眼区域，即图像的右半区域
                               img_facehalf_right截取右眼区域，在图像的左半区域
                               img_facehalf_bottom是截取人脸识别到区域下半部分'''
                left_gray = face_gray[y + eh1:y + eh2, x + ew2:x + ew3]
                right_gray = face_gray[y + eh1:y + eh2, x + ew1:x + ew2]
                mouth_gray = face_gray[y + mh1:y + mh2, x:x + w]

                left_color = face_color[y + eh1:y + eh2, x + ew2:x + ew3]
                right_color = face_color[y + eh1:y + eh2, x + ew1:x + ew2]
                eye_color = face_color[y + eh1:y + eh2, x + ew1:x + ew3]
                mouth_color = face_color[y + mh1:y + mh2, x:x + w]

                cv2.imshow("eye", cv2.cvtColor(eye_color, cv2.COLOR_BGR2GRAY))
                eye_to_save = cv2.resize(eye_color, (30, 100), interpolation=cv2.INTER_AREA)
                # cv2.imwrite('%s/%d.jpg' % (folder_path_eye, num), eye_to_save)

                cv2.imshow("mouth", cv2.cvtColor(mouth_color, cv2.COLOR_BGR2GRAY))
                mouth_to_save = cv2.resize(mouth_color, (50, 100), interpolation=cv2.INTER_AREA)
                # cv2.imwrite('%s/%d.jpg' % (folder_path_mouth, num), mouth_to_save)

                # 画出左右眼和人脸下部分区域
                '''矩形画出区域 rectangle参数（图像，左顶点坐标(x,y)，右下顶点坐标（x+w,y+h），线条颜色，线条粗细）'''
                cv2.rectangle(face_color, (x + ew1, y + eh1), (x + ew2, y + eh2), (255, 0, 0), 2)
                cv2.rectangle(face_color, (x + ew2, y + eh1), (x + ew3, y + eh2), (255, 0, 0), 2)
                cv2.rectangle(face_color, (x + mw1, y + mh1), (x + mw2, y + mh2), (255, 0, 0), 2)

                # 检测器识别左眼
                # for small
                lefteyes = lefteye_classifier.detectMultiScale(left_gray, scaleFactor=1.05, minNeighbors=1,
                                                               minSize=(20, 20), maxSize=(80, 80),
                                                               flags=cv2.CASCADE_SCALE_IMAGE)
                # for big
                # faceRects_lefteye = classifier_eye.detectMultiScale(img_facehalf, scaleFactor=1.3, minNeighbors=16, minSize=(40, 40),maxSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE)

                # 检测到左眼后循环
                if len(lefteyes) > 0:
                    for lefteye in lefteyes:
                        xl1, yl1, wl1, hl1 = lefteye
                        # 画出左眼区域
                        cv2.rectangle(left_color, (xl1, yl1), (xl1 + wl1, yl1 + hl1), (0, 255, 0), 2)

                # 检测器识别右眼
                # for small
                righteyes = righteye_classifier.detectMultiScale(right_gray, scaleFactor=1.05, minNeighbors=1,
                                                                 minSize=(20, 20), maxSize=(80, 80),
                                                                 flags=cv2.CASCADE_SCALE_IMAGE)
                # for big
                # faceRects_righteye = classifier_eye.detectMultiScale(img_facehalf, scaleFactor=1.3, minNeighbors=16, minSize=(40, 40),maxSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE)

                # 检测到右眼后循环
                if len(righteyes) > 0:
                    for righteye in righteyes:
                        xr1, yr1, wr1, hr1 = righteye
                        # 画出右眼区域
                        cv2.rectangle(right_color, (xr1, yr1), (xr1 + wr1, yr1 + hr1), (0, 255, 0), 2)

                # 嘴巴检测器
                # for small
                mouths = mouth_classifier.detectMultiScale(mouth_gray, scaleFactor=1.05, minNeighbors=10,
                                                           minSize=(5, 5), maxSize=(80, 80),
                                                           flags=cv2.CASCADE_SCALE_IMAGE)
                # for big
                # faceRects_mouth=classifier_mouth.detectMultiScale(img_facehalf_bottom,scaleFactor=1.3,minNeighbors=10,minSize=(40,40),maxSize=(80,80),flags=cv2.CASCADE_SCALE_IMAGE)

                if len(mouths) > 0:
                    for mouth in mouths:
                        xm1, ym1, wm1, hm1 = mouth
                        cv2.rectangle(mouth_color, (xm1, ym1), (xm1 + wm1, ym1 + hm1), (0, 0, 255), 2)

                num += 1
                cv2.putText(frame_rotated, 'num:%d' % num, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),
                            2)

                #旋转帧至原来位置并修剪黑色填充物
                frame_resized = ut.trim(ut.rotate_image(frame_rotated, rotation * (-1)), frame_scale)
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

    cv2.putText(frame_resized, "Press 'ESC/Q/q' to quit.", (5, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv2.imshow("Real Time Facial Recognition", frame_resized)

    #获取下一帧
    ret, frame = webcam.read()

    # 超过指定最大保存数量结束程序
    if num > num_of_save:
        break

    # video.write(frame_resized)

webcam.release()
cv2.destroyAllWindows()