import os
import numpy
from PIL import Image, ImageDraw
import cv2

#获取外接摄像头
webcam = cv2.VideoCapture(0)

#获取摄像头返回的宽和高,确定保存视频的格式
# ''' cv.VideoWriter参数（视频存放路径，视频存放格式，fps帧率，视频宽高）
#     注意点1：OpenCV只支持avi的格式，而且生成的视频文件不能大于2GB，而且不能添加音频
#     注意点2：若填写的文件名已存在，则该视频不会录制成功，但可正常使用'''
# size = (int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5), int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video = cv2.VideoWriter(r"C:\Users\RongYue\Desktop\FatigueTests\eye_mouth_openCV.avi", fourcc, 5, size)

#正脸和侧脸识别
face_classifier = cv2.CascadeClassifier(r"C:\Program Files\opencv\opencv-master\data\haarcascades\haarcascade_frontalface_alt.xml")
sideFace_classifier = cv2.CascadeClassifier(r"C:\Program Files\opencv\opencv-master\data\haarcascades\haarcascade_profileface.xml")
#左眼和右眼识别
lefteye_classifier = cv2.CascadeClassifier(r"C:\Program Files\opencv\opencv-master\data\haarcascades\haarcascade_lefteye_2splits.xml")
righteye_classifier = cv2.CascadeClassifier(r"C:\Program Files\opencv\opencv-master\data\haarcascades\haarcascade_righteye_2splits.xml")
#嘴巴识别
mouth_classifier=cv2.CascadeClassifier(r"C:\Program Files\opencv\opencv_contrib-master\modules\face\data\cascades\haarcascade_mcs_mouth.xml")

#检测是否摄像头正常打开:成功打开时，isOpened返回ture
while (webcam.isOpened()):
    '''第一个参数ret的值为True或False，代表有没有读到图片
           第二个参数是frame，是当前截取一帧的图片
        '''
    ret, frame = webcam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    #检测
    '''detectMultiScale参数
    （要操作的图像img；
    每次缩小图像的比例scale——合理范围1.1~1.4 尺度越大越容易遗漏检测对象但检测速度加快，尺度越小检测越细致速度越慢；
    投票数minNeighbors——默认0，增加投票数，检测条件越苛刻，越容易滤除误检对象；
    匹配物体的大小范围——minSize(像素*像素)~maxSize(像素*像素)，要使得图像落在检测器的范围内）'''
    # for small
    faces = face_classifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=4,minSize=(60,60),maxSize=(300,300),flags=cv2.CASCADE_SCALE_IMAGE)
    # 如果没有正脸，试试侧脸
    faces = faces if len(faces) else sideFace_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(30, 30), flags=cv2.IMREAD_GRAYSCALE)
    print("a: ", faces)

    # 检测到人脸
    if len(faces) > 0:

        # faceRects = faceRects_face[np.lexsort(-faceRects.T)]#按最后一列h逆序排序
        # print("b: ", faceRects)
        # x, y, w, h = faceRects[0]
        # print("c: ",x,y,w,h)

        faces = faces[(faces[:, 2] * faces[:, 3]).argsort()]  # 按照人脸面积w*h从小到大排序
        print("b: ", faces)
        x, y, w, h = faces[-1]  # 取最大人脸
        print("c: ", x, y, w, h)

        cv2.rectangle(frame, (x, y), (x + w, y+h), (255, 0, 0), 2)
        face_gray = cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_color = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


        # 眼睛位于人脸高度0.2h~0.5h，宽度0.13w~0.87W，对此位置定位以精确识别眼睛
        eh1 = int(h * 0.25)
        eh2 = int(h * 0.5)
        ew1 = int(w * 0.13)
        ew2 = int(w * 0.5)
        ew3 = int(w * 0.87)
        # 同样的，截取人脸区域下半部分，以精确识别嘴巴的位置
        mh1 = int(h * 0.7)
        mh2 = int(h * 1)
        mw1 = int(w * 0.25)
        mw2 = int(w * 0.75)

        '''img获取坐标为，【y,y+h之间（竖）：x,x+w之间(横)范围内的数组】
                       img_facehalf_left 截取左眼区域，即图像的右半区域
                       img_facehalf_right截取右眼区域，在图像的左半区域
                       img_facehalf_bottom是截取人脸识别到区域下半部分'''
        left_gray = face_gray[y + eh1:y +eh2, x + ew2:x + ew3]
        right_gray=face_gray[y + eh1:y + eh2,x + ew1:x + ew2]
        bottom_gray = face_gray[y+mh1:y+mh2, x:x + w]

        left_color = face_color[y + eh1:y + eh2, x + ew2:x + ew3]
        right_color = face_color[y + eh1:y + eh2, x + ew1:x + ew2]
        bottom_color = face_color[y + mh1:y + mh2, x:x + w]

        # 画出左右眼和人脸下部分区域
        '''矩形画出区域 rectangle参数（图像，左顶点坐标(x,y)，右下顶点坐标（x+w,y+h），线条颜色，线条粗细）'''
        cv2.rectangle(face_color, (x + ew1, y + eh1), (x + ew2, y + eh2), (255, 0, 0), 2)
        cv2.rectangle(face_color, (x + ew2, y + eh1), (x + ew3, y + eh2), (255, 0, 0), 2)
        cv2.rectangle(face_color, (x + mw1, y + mh1), (x + mw2, y + mh2), (255, 0, 0), 2)

        # 检测器识别左眼
        #for small
        lefteyes = lefteye_classifier.detectMultiScale(left_gray, scaleFactor=1.05, minNeighbors=1,minSize=(20,20),maxSize=(80,80),flags=cv2.CASCADE_SCALE_IMAGE)
        # for big
        # faceRects_lefteye = classifier_eye.detectMultiScale(img_facehalf, scaleFactor=1.3, minNeighbors=16, minSize=(40, 40),maxSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE)

        # 检测到左眼后循环
        if len(lefteyes) > 0:
            for lefteye in lefteyes:
                xl1, yl1, wl1, hl1 = lefteye
                # 画出左眼区域
                cv2.rectangle(left_color, (xl1, yl1), (xl1 + wl1, yl1 + hl1), (0, 255, 0), 2)

        # 检测器识别右眼
        #for small
        righteyes = righteye_classifier.detectMultiScale(right_gray, scaleFactor=1.05, minNeighbors=1,minSize=(20,20),maxSize=(80,80),flags=cv2.CASCADE_SCALE_IMAGE)
        # for big
        # faceRects_righteye = classifier_eye.detectMultiScale(img_facehalf, scaleFactor=1.3, minNeighbors=16, minSize=(40, 40),maxSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE)

        # 检测到右眼后循环
        if len(righteyes) > 0:
            for righteye in righteyes:
                xr1, yr1, wr1, hr1 = righteye
                # 画出右眼区域
                cv2.rectangle(right_color, (xr1, yr1), (xr1 + wr1, yr1 + hr1), (0, 255, 0), 2)


        # 嘴巴检测器
        #for small
        mouths=mouth_classifier.detectMultiScale(bottom_gray,scaleFactor=1.05,minNeighbors=10,minSize=(5,5),maxSize=(80,80),flags=cv2.CASCADE_SCALE_IMAGE)
        #for big
        # faceRects_mouth=classifier_mouth.detectMultiScale(img_facehalf_bottom,scaleFactor=1.3,minNeighbors=10,minSize=(40,40),maxSize=(80,80),flags=cv2.CASCADE_SCALE_IMAGE)

        if len(mouths) > 0:
            for mouth in mouths:
                xm1, ym1, wm1, hm1 = mouth
                cv2.rectangle(bottom_color, (xm1,ym1), (xm1+ wm1, ym1 + hm1),(0, 0, 255), 2)

        # video.write(frame)
    # 显示图片，标题名字为video
    cv2.imshow('video', frame)
    #调整窗口大小video为64*64
    cv2.resizeWindow('video',540,480)
    # 检测到键盘输入ESC Q q，退出循环
    key = cv2.waitKey(1)
    if key in [27, ord('Q'), ord('q')]:
        break

# video.release()
#不再录制视频
webcam.release()
#释放摄像头
cv2.destroyAllWindows()
#关闭所有窗口显示