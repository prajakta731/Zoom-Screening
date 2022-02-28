import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time 
from time import sleep
from mss import mss
from PIL import Image
#time.sleep(300)


face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl=['blink','Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
# cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
count2=0
score=0
score2=0
thicc=2
rpred=[99]
lpred=[99]
rpred2=[99]
lpred2=[99]
p1Playing = True
p2Playing = True



def blinkPred(count,score,thicc,rpred,lpred):
    ret, frame = cap.read()
    height,width = frame.shape[:2]


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        #sleep(0)
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye= r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        if(rpred[0]==1):
            lbl='Open'
        if(rpred[0]==0):
            lbl='blink'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        #sleep(0)
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            lbl='Open'
        if(lpred[0]==0):
            lbl='blink'
        break

    if(rpred[0]==1 and lpred[0]==1):
        score=score+1
        cv2.putText(frame,"open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    #if(rpred[0]==0 or lpred[0]==0):
    else:
        #score=score>1
        cv2.putText(frame,"blink",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        # break
        # exit()
        pass
        cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        
    if(score<0):
        score=0
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>5):
        #person is feeling sleepy so we beep the alarm
        #cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # break
        # exit()
        cv2.destroyAllWindows()
        return False
        pass
# while(True):
   
    # val=blinkPred(count,score,thicc,rpred,lpred)
    # if not val:
        # break

# cap.release()
# cv2.destroyAllWindows()

# TIMER

with open("timer.txt", 'w') as f:
    f.seek(0)
    f.write("Starting...")
sleep(2)
with open("timer.txt", 'w') as f:
    f.seek(0)
    f.write("3")
sleep(2)
with open("timer.txt", 'w') as f:
    f.seek(0)
    f.write("2")
sleep(2)
with open("timer.txt", 'w') as f:
    f.seek(0)
    f.write("1")
sleep(2)
with open("timer.txt", 'w') as f:
    f.seek(0)
    f.write("Begin...")


mon = {'left': 682, 'top': 159, 'width': 680, 'height': 385}
mon2 = {'left': 0, 'top': 159, 'width': 680, 'height': 385}

with mss() as sct:
    while True:
        screenShot = sct.grab(mon)
        screenShot2 = sct.grab(mon2)

        img = Image.frombytes(
            'RGB', 
            (screenShot.width, screenShot.height), 
            screenShot.rgb, 
        )
        img2 = Image.frombytes(
            'RGB', 
            (screenShot2.width, screenShot2.height), 
            screenShot2.rgb, 
        )

        im = np.array(screenShot)
        im = np.flip(im[:, :, :3], 2)
        frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # 2

        im2 = np.array(screenShot2)
        im2 = np.flip(im2[:, :, :3], 2)
        frame2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)  # 2

        height,width = frame.shape[:2]
        height2,width2 = frame2.shape[:2]


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)

        faces2 = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
        left_eye2 = leye.detectMultiScale(gray2)
        right_eye2 = reye.detectMultiScale(gray2)

        cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

        for (x,y,w,h) in right_eye:
            r_eye=frame[y:y+h,x:x+w]
            count=count+1
            #sleep(0)
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255
            r_eye= r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = model.predict_classes(r_eye)
            if(rpred[0]==1):
                lbl='Open'
            if(rpred[0]==0):
                lbl='blink'
            break
        for (x,y,w,h) in right_eye2:
            r_eye2=frame2[y:y+h,x:x+w]
            count2=count2+1
            r_eye2 = cv2.cvtColor(r_eye2,cv2.COLOR_BGR2GRAY)
            r_eye2 = cv2.resize(r_eye2,(24,24))
            r_eye2= r_eye2/255
            r_eye2= r_eye2.reshape(24,24,-1)
            r_eye2 = np.expand_dims(r_eye2,axis=0)
            rpred2 = model.predict_classes(r_eye2)

        for (x,y,w,h) in left_eye:
            l_eye=frame[y:y+h,x:x+w]
            count=count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = model.predict_classes(l_eye)
            if(lpred[0]==1):
                lbl='Open'
            if(lpred[0]==0):
                lbl='blink'
            break
        for (x,y,w,h) in left_eye2:
            l_eye2=frame2[y:y+h,x:x+w]
            count2=count2+1
            l_eye2 = cv2.cvtColor(l_eye2,cv2.COLOR_BGR2GRAY)
            l_eye2 = cv2.resize(l_eye2,(24,24))
            l_eye2= l_eye2/255
            l_eye2=l_eye2.reshape(24,24,-1)
            l_eye2 = np.expand_dims(l_eye2,axis=0)
            lpred2 = model.predict_classes(l_eye2)

        if(p1Playing):
            if(rpred[0]==1 and lpred[0]==1):
                score=score+1
                with open('1.txt', 'w') as f:
                    f.seek(0)
                    f.write("Score: " + str(score))
                cv2.putText(frame,"open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            else:
                cv2.putText(frame,"blink",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
                with open('1.txt', 'w') as f:
                    f.seek(0)
                    f.write("End Score: " + str(score))
                # break
                print("p1 lost")
                p1Playing = False
                cv2.putText(frame,'end Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

        if(p2Playing):
            if(rpred2[0]==1 and lpred2[0]==1):
                score2=score2+1
                with open('2.txt', 'w') as f:
                    f.seek(0)
                    f.write("Score: " + str(score2))
                cv2.putText(frame2,"open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            else:
                cv2.putText(frame2,"blink",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
                with open('2.txt', 'w') as f:
                    f.seek(0)
                    f.write("End Score: " + str(score2))
                # break
                print("p2 lost")
                p2Playing = False
                cv2.putText(frame2,'end Score:'+str(score2),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            
        if (p1Playing):
            if(score<0):
                score=0
            cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        if (p2Playing):
            if(score2<0):
                score2=0
            cv2.putText(frame2,'Score:'+str(score2),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

        # cv2.imshow('frame',frame)
        # cv2.imshow('frame2',frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # break
            # exit()
            break
        if cv2.waitKey(33) & 0xFF in (
            ord('q'), 
            27, 
        ):
            break
    cv2.destroyAllWindows()