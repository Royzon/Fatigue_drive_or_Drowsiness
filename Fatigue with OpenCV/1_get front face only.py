import cv2
from time import clock


def draw_rects(img,rects,color):
    for x,y,w,h in rects:
        cv2.rectangle(img,(x - 10, y - 10), (x + w + 10, y + h + 10),color,2)


def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)

    # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)

    # 告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier(
        r"C:\Program Files\opencv\opencv-master\data\haarcascades\haarcascade_frontalface_alt2.xml")

    # 识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)

    num = 0
    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像

        t=clock()

        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        print("a: ", faceRects)

        if len(faceRects) > 0:  # 大于0则检测到人脸

            # faceRects = faceRects_face[np.lexsort(-faceRects.T)]#按最后一列h逆序排序
            # print("b: ", faceRects)
            # x, y, w, h = faceRects[0]
            # print("c: ",x,y,w,h)

            faceRects = faceRects[(faceRects[:, 2] * faceRects[:, 3]).argsort()]  # 按照人脸面积w*h从小到大排序
            print("b: ", faceRects)
            x, y, w, h = faceRects[-1]  # 取最大人脸
            print("c: ", x, y, w, h)

            # 画出矩形框
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

            #显示并保存当前帧
            image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
            cv2.imshow('face',image)

            dt=clock()-t

            img_name = '%s/%d.jpg' % (path_name, num)
            # cv2.imwrite(img_name, image)

            num += 1
            if num > (catch_pic_num):  # 如果超过指定最大保存数量退出循环
                break

            # 显示时间和当前捕捉到了多少人脸图片
            cv2.putText(frame, 'time: %.1f ms' %(dt*1000), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)

            # 超过指定最大保存数量结束程序
        if num > (catch_pic_num):
            break

        # 显示图像
        cv2.imshow(window_name, frame)

        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    CatchPICFromVideo("get_face", 0, 500, r'E:\HASCO\project\code\FatigueTests\1_OpenCV')