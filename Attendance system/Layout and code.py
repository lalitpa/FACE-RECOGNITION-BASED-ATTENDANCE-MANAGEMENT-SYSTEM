import tkinter as tk
from tkinter import Message,Text
import cv2 , os
import shutil
import csv
import numpy as np
from PIL import Image ,ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

window=tk.Tk()
window.title("Face Recognition")
window.geometry('1280x720')
dialog_title="Quit"
dialog_text="Are you sure??"
window.configure(background='cyan')
window.grid_rowconfigure(0,weight=1)
window.grid_columnconfigure(0,weight=1)

message=tk.Label(window,padx=0.5,text='FACE RECOGNITION BASED ATTENDANCE SYSTEM',bg='cyan',fg='goldenrod4',width=50,height=4,font=('times',30,'bold '))
message.place(x=100,y=-40)
lb1=tk.Label(window,text="Enter Id",width=20,height=2,bg='purple3',fg='white' ,font=('times',15,' bold'))
lb1.place(x=100,y=150)
txt=tk.Entry(window,width=20,bg='white',font=('times',25,' bold'))
txt.place(x=450,y=150)

lb2=tk.Label(window,text="Enter Name",width=20,height=2,bg='purple3',fg='white' ,font=('times',15,' bold'))
lb2.place(x=100,y=250)
txt2=tk.Entry(window,width=20,bg='white',font=('times',25,' bold'))
txt2.place(x=450,y=250)

lb3=tk.Label(window,text="NOTIFICATION",width=20,height=2,bg='blue2',fg='lightyellow' ,font=('times',15,' bold'))
lb3.place(x=100,y=340)

message=tk.Label(window,text='',bg='white',fg='red',width=35,height=1, activebackground='yellow',font=('times',30,'bold'))
message.place(x=450,y=340)


lb4=tk.Label(window,text="ATTENDANCE",width=20,height=4,bg='cyan',fg='blue2' ,font=('times',30,' bold'))
lb4.place(x=-95,y=550)

message2=tk.Label(window,text='',bg='white',fg='red',width=30,height=2, activebackground='yellow',font=('times',30,'bold'))
message2.place(x=300,y=600)


def clear():
    txt.delete(0,'end')
    res=''
    message.configure(text=res)
def clear2():
    txt2.delete(0,'end')
    res=''
    message.configure(text=res)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError,ValueError):
        pass
    return False

#take image method

def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():        
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                 
                sampleNum=sampleNum+1
                
                cv2.imwrite("TrainingImages\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                
                cv2.imshow('frame',img)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImages")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainer.yml")
    res = "Image Trained"#+",".join(str(f) for f in Id)
    message.configure(text= res)

def getImagesAndLabels(path):
    
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    
    
    
    faces=[]
    
    Ids=[]
    
    for imagePath in imagePaths:
        
        pilImage=Image.open(imagePath).convert('L')
        
        imageNp=np.array(pilImage,'uint8')
        
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
       
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainer.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im)
        if (cv2.waitKey(1)==ord('q')):
          break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)
    res=attendance
    message2.configure(text= res)





clearButton=tk.Button(window,text='Clear', command=clear,bg='grey',fg='lightyellow',width=10,height=1, activebackground='red',font=('times',30,'bold'))
clearButton.place(x=950,y=140)

clearButton2=tk.Button(window,text='Clear', command=clear2,bg='grey',fg='lightyellow',width=10,height=1, activebackground='red',font=('times',30,'bold'))
clearButton2.place(x=950,y=240)

takeImg=tk.Button(window,text='UPLOAD IMAGE',command=TakeImages,bg='goldenrod2',fg='lightyellow',width=15,height=2, activebackground='red',font=('times',30,'bold'))
takeImg.place(x=10,y=440)

trainImg=tk.Button(window,text='TRAINER',command=TrainImages,bg='goldenrod2',fg='lightyellow',width=10,height=2, activebackground='red',font=('times',30,'bold'))
trainImg.place(x=410,y=440)

trackImg=tk.Button(window,text='MARK YOUR ATTENDANCE',command=TrackImages,bg='goldenrod2',fg='lightyellow',width=25,height=2, activebackground='red',font=('times',30,'bold'))
trackImg.place(x=710,y=440)

quitwindow=tk.Button(window,text='Quit',command=window.destroy,bg='red3',fg='lightyellow',width=10,height=1, activebackground='white',font=('times',30,'bold'))
quitwindow.place(x=1050,y=600)


window.mainloop()


