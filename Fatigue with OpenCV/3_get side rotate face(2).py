import cv2
import numpy as np
from math import *

#获取外接摄像头
# video='with_glass.mp4'
# cap = cv2.VideoCapture(video)
cap = cv2.VideoCapture(0)

#定义正脸和侧脸检测器
face_cascade = cv2.CascadeClassifier(r"C:\Program Files\opencv\opencv-master\data\haarcascades\haarcascade_frontalface_alt.xml")
sideFace_cascade = cv2.CascadeClassifier(r"C:\Program Files\opencv\opencv-master\data\haarcascades\haarcascade_profileface.xml")

num=1
catch_pic_num=200
rotate_num = 0  # 旋转的次数
degree = 60  # 每次旋转的角度

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

#检测是否摄像头正常打开:成功打开时，isOpened返回ture
while (cap.isOpened()):
    ret, frame = cap.read()
    '''第一个参数ret的值为True或False，代表有没有读到图片
       第二个参数是frame，是当前截取一帧的图片
    '''
    rotate_num += 1

    frame_gamma = gamma_correction_auto(frame, equalizeHist=False)
    frame_gray = cv2.cvtColor(frame_gamma, cv2.COLOR_BGR2GRAY)
    frame_Hist = cv2.equalizeHist(frame_gray)
    frame_filter = cv2.bilateralFilter(frame_Hist, 3, 50, 50)

    #检测器：detectMultiScale参数（图像，每次缩小图像的比例，匹配成功所需要的周围矩形框的数目，检测的类型，匹配物体的大小范围）
    # faces = classifier_face.detectMultiScale(frame, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, (20, 20))
    # 检测正脸
    faces = face_cascade.detectMultiScale(
        frame_filter,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(5, 5),
        flags=cv2.IMREAD_GRAYSCALE
    )
    # 如果没有正脸，试试侧脸
    faces = faces if len(faces) else sideFace_cascade.detectMultiScale(
        frame_filter,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.IMREAD_GRAYSCALE
    )
    print("Found {} faces!".format(len(faces)))
    # 检测到人脸
    if len(faces) > 0:

        img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_filter = cv2.equalizeHist(img1)

        # 检测器：detectMultiScale参数（图像，每次缩小图像的比例，匹配成功所需要的周围矩形框的数目，检测的类型，匹配物体的大小范围）
        # faces = classifier_face.detectMultiScale(frame, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, (20, 20))
        # 检测正脸
        faces = face_cascade.detectMultiScale(
            frame_filter,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(5, 5),
            flags=cv2.IMREAD_GRAYSCALE
        )
        # 如果没有正脸，试试侧脸
        faces = faces if len(faces) else sideFace_cascade.detectMultiScale(
            frame_filter,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.IMREAD_GRAYSCALE
        )

        print("a: ", faces)

        if len(faces) > 0:

            # faces = faces[np.lexsort(-faces.T)]#按最后一列h逆序排序
            # print("b: ", faces)
            # x, y, w, h = faces[0]
            # print("c: ",x,y,w,h)

            faces = faces[(faces[:, 2] * faces[:, 3]).argsort()]  # 按照人脸面积w*h从小到大排序
            print("b: ", faces)
            x, y, w, h = faces[-1]#取最大人脸
            print("c: ", x, y, w, h)

            img_face = frame[y - 10: y + h + 50, x - 10: x + w + 10]
            print(img_face.shape)
            # cv2.imwrite('%s/%d.jpg' % (r'E:\HASCO\project\code\FatigueTests\data\faces', num), img_face)

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

            # 画出左右眼和人脸下部分区域
            '''矩形画出区域 rectangle参数（图像，左顶点坐标(x,y)，右下顶点坐标（x+w,y+h），线条颜色，线条粗细）'''
            cv2.rectangle(frame, (x + ew1, y + eh1), (x + ew2, y + eh2), (255, 0, 0), 2)
            cv2.rectangle(frame, (x + ew2, y + eh1), (x + ew3, y + eh2), (255, 0, 0), 2)
            cv2.rectangle(frame, (x + mw1, y + mh1), (x + mw2, y + mh2), (255, 0, 0), 2)

            '''img获取坐标为，【y,y+h之间（竖）：x,x+w之间(横)范围内的数组】
                           img_facehalf是截取人脸识别到区域上半部分
                           img_facehalf_bottom是截取人脸识别到区域下半部分
                           为了准确切割可适当缩小范围'''
            img_lefteye = frame[y + eh1:y +eh2, x + ew2:x + ew3]
            img_righteye = frame[y + eh1:y + eh2,x + ew1:x + ew2]
            img_mouth = frame[y+mh1:y+mh2, x:x + w]

            # 预处理
            # img_lefteye = cv2.equalizeHist(img_lefteye)
            img_lefteye = cv2.medianBlur(img_lefteye, 7)
            img_lefteye = cv2.medianBlur(img_lefteye, 3)
            # img_righteye = cv2.equalizeHist(img_righteye)
            img_righteye = cv2.medianBlur(img_righteye, 7)
            img_righteye = cv2.medianBlur(img_righteye, 3)

            kernel = np.ones((5, 5), np.float32) / 25
            img_mouth = cv2.filter2D(img_mouth, -1, kernel)

            # cv2.imwrite('%s/%d.jpg' % (r'E:\HASCO\project\code\FatigueTests\data\left_eyes', num), img_lefteye)
            # cv2.imwrite('%s/%d.jpg' % (r'E:\HASCO\project\code\FatigueTests\data\right_eyes', num), img_righteye)
            # cv2.imwrite('%s/%d.jpg' % (r'E:\HASCO\project\code\FatigueTests\data\mouths', num), img_mouth)

            # 指定最大保存数量
            num += 1
            cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)

        # 超过指定最大保存数量结束程序
        if num > (catch_pic_num):
            break


    else:  # 如果检测不到人脸
        if rotate_num == int(360 / degree):  # 判断是否旋转完360度
            print('frame {}: after rotate {} degree, No face is detected!'.format(frame, degree * (rotate_num - 1)))
            break
            # 旋转60度
        rows, cols, channel = frame.shape
        # 为了旋转之后不裁剪原图，计算旋转后的尺寸
        rowsNew = int(cols * fabs(sin(radians(degree))) + rows * fabs(cos(radians(degree))))
        colsNew = int(rows * fabs(sin(radians(degree))) + cols * fabs(cos(radians(degree))))
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)  # 旋转60度的仿射矩阵
        M[0, 2] += (colsNew - cols) / 2
        M[1, 2] += (rowsNew - rows) / 2
        frame = cv2.warpAffine(frame, M, (colsNew, rowsNew), borderValue=(255, 255, 255))  # 旋转60度，得到新图片

    #显示图片，标题名字为video
    cv2.imshow('video', frame)
    #调整窗口大小video为64*64
    cv2.resizeWindow('video',540,480)

    #检测到键盘输入q，退出循环
    if cv2.waitKey(1) in [27,ord('Q'),ord('q')]:
        break

#释放摄像头
cap.release()
#关闭所有窗口显示
cv2.destroyAllWindows()
