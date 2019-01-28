import pdb
import numpy as np
import cv2
import datetime
import time
import sys, select, os
#import pyautogui



timen=0

first=0
from Tkinter import *

def guicount(a,b):
  #from Tkinter import *
  count=a
  root = Tk()

  root.title("Clicks Counter")
  root.geometry("800x800")
  if b<10:
   mode=1
  if b>10 and b<20:
   mode=2
  if b>20: #and b<25
   mode=3
  #if b>25
  
  app=Frame(root)
  app.grid()
  if mode ==1:
   y="\n\n\n No of blinks detected in the session  "+str(count)+"\n    Average no. of blinks per minute ="+str(b)+"\nThe person is focused  :)";
  if mode ==2:
   y="\n\n\n No of blinks detected in the session  "+str(count)+"\n    Average no. of blinks per minute ="+str(b)+"\nThe Person is normal ";
  if mode ==3:
   y="\n\n\n No of blinks detected in the session  "+str(count)+"\n    Average no. of blinks per minute ="+str(b)+"\nThe person is under stress or in a conversation";
   
 # y1="\n Average no. of blinks per minute ="+str(b);
  label=Label(app,text=y)
  #label=Label(app,text=y1)
  label.grid()
  
  root.mainloop()

T1=time.time()


#import video
time1=0
time2=0
time3=0
flag=0
count_blink=0
#cap.set(CV_CAP_PROP_FPS, 10);
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
eye_cascade1 = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
eye_cascade2 = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('fcomtest3.webm')
#cap.set(cv2.cv.CV_CAP_PROP_FPS, 10);

#file = open("testfile.txt","w")
count=0 
while 1:
    ret, img = cap.read()

   # fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    #print fps
   # print "is the frame rate"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    # pdb.set_trace()
    PUP = cv2.CascadeClassifier('haarcascade_eye.xml')
    neye = PUP.detectMultiScale(gray,1.3,5)
    pupilFrame = gray
    PupilO = gray
    windowClose = np.ones((5,5),np.uint8)
    windowOpen = np.ones((2,2),np.uint8)
    windowErode = np.ones((2,2),np.uint8)
    #pdb.set_trace()
    for x,y,w,h in faces:
        #print "asas"
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        #pdb.set_trace()
        for x1,y1,w1,h1 in neye:
            cv2.rectangle(roi_color, (x1,y1), ((x1+w1),(y1+h1)), (0,0,255),1)
            cv2.line(roi_color, (x1,y1), ((x1+w,y+h1)), (0,0,255),1)
            cv2.line(roi_color, (x1+w1,y1), ((x1,y1+h1)), (0,0,255),1)
            cv2.circle(img, (x1+w1/2,y1+h1/2),5, (255,255,100), -5)#, lineType=8, shift=0)
            posix=x1+w1/2
            posiy=y1+h1/2
            pupilFrame = cv2.equalizeHist(gray[y1+(h1/4):(y1+h1), x1:(x1+w1)])
            pupilO = pupilFrame
            ret, pupilFrame = cv2.threshold(pupilFrame,55,255,cv2.THRESH_BINARY)		#50 ..nothin 70 is better
        file = open("testfile.txt","a")
        file.write(str(x1+w1/2))
        file.write("---")
        file.write(str(y1+h1/2))
        file.write("-----------------------\n")
        file.close()
        #print(1)
       # print cy
	 #  pyautogui.moveTo(cx*5, cy*3)
		#show picture
        if posix > 340 and posix < 395 :        #(gap)>4 and gap<10 :
          cv2.putText(img, "looking straight", (150,300), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,250), thickness=3)#linetype=cv2.CV_AA)
        elif posix < 340: #(gap)>10:
          cv2.putText(img, "looking Right", (150,300), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,250), thickness=3)#linetype=cv2.CV_AA)
        elif posix>395:# (gap)<10
          cv2.putText(img,"looking Left", (150,300), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,250), thickness=3)#linetype=cv2.CV_AA)
        eyes1 = eye_cascade1.detectMultiScale(roi_gray)

        eyes2 = eye_cascade2.detectMultiScale(roi_gray)
        for ex,ey,ew,eh in eyes1:
          cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
        for ex,ey,ew,eh in eyes2:
          cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


   # eye=cv2.GetSubRect(img,(ex,ey,ex+ew,ey+eh))
    #cv2.imshow('img',img)
   # cv2.imshow('gray',eye)
    # Otsu's thresholding after Gaussian filtering
   # blur = cv2.GaussianBlur(roi_gray,(5,5),0)
 #   ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  #  cv2.imshow('BW',th3)
    ne1=len(eyes1)
    ne2=len(eyes2)
    #print ne1-left
   #date_time = datetime.datetime.strptime(date_time_string, '%Y-%m-%d %H:%M')
    
    if ne1>0 and ne2>0 :
     if ne1==1 and ne2==1 :
       
       t1=time.time()
       if first==0:
        timen=t1
       else:
        time1=t1
       first=first+1
       count=count+1
       flag=1
      # print "Both Closed!\n at "
     # file.write("\nBoth Closed!\n ")


       #now = datetime.datetime.now()
      # t2=time.time()
     #  time2=t2
         #print (t2-t1)

      # print now
       #print "\n--------------"
       #print count
       #print "--------------\n"
    

     elif ne2==1 and ne1!=1 :
      a=0
     # print "Right Closed"
     elif ne2!=1 and ne1==1:
      b=0
     # print "left closed"
     else:
      t3=time.time()
      time3=t3 
     # print "Both Open"
      #print t1
      if flag==1:
       flag=0
       print "closed for :"
       print time3-timen
       if (time3-timen)>4:
        #import winsound
        #Freq = 2500 # Set Frequency To 2500 Hertz
        #Dur = 1000 # Set Duration To 1000 ms == 1 second
        #winsound.Beep(Freq,Dur)
        #os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % ( 2, 2500))
        os.system("/usr/bin/canberra-gtk-play --id='bell'")
        os.system("/usr/bin/canberra-gtk-play --id='bell'")
        os.system("/usr/bin/canberra-gtk-play --id='bell'")
        os.system("/usr/bin/canberra-gtk-play --id='bell'")
        os.system("/usr/bin/canberra-gtk-play --id='bell'")
        os.system("/usr/bin/canberra-gtk-play --id='bell'")

       count_blink=count_blink+1
       timen=time.time()




       os.system('cls' if os.name == 'nt' else 'clear')
      # T2=time.time()
    '''
       if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
         line = raw_input()
         T2=time.time()
         mins=(T2-T1)/60
         bpm=count_blink/mins
         guicount(count_blink,bpm)
         break
	     # f = open("clicks.txt", "w")
              #f.write( str(recordedEvents)  )      # str() converts to string
 
        # break

      '''


    #A1=l1*w1
   # cv2.putText(img, "your_string", (100,300), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,250), thickness=3)#linetype=cv2.CV_AA)
    cv2.imshow('img',img)

    #cv2.imshow('000',pupilO)
    #cv2.imshow('000',pupilFrame)
    #cv2.imshow('gray',gray)
    k = cv2.waitKey(30) #& 0xff
    if k == 27:
      # file.close()
     cap.release()
     cv2.destroyAllWindows()
     T2=time.time()
     mins=(T2-T1)/60
     bpm=count_blink/mins
     guicount(count_blink,bpm)
     cap.release()
     cv2.destroyAllWindows()
	

    # guicount(count_blink)
     break
    #if input()=='e' :
       #file.close()
       # break

cap.release()
cv2.destroyAllWindows()
