from tkinter import *
from tkinter import messagebox as msg

from tkinter import filedialog
from PIL import Image,ImageTk,ImageDraw
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# ecompile loaded model
loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


root=Tk()
bgclr='powder blue'
root.state('zoomed')
root.update()
root.configure(bg=bgclr)
root.resizable(width=False,height=False)
lbl_head=Label(root,bg='orange',text='Gender Prediction',font=('verdna',45,'bold'))
lbl_head.pack(side='top',anchor='c')


# def use_video():


def use_image(frm):
    h=3
    login_frm=Frame(root,bg=bgclr)
    login_frm.place(x=0,y=100,width=root.winfo_width(),height=root.winfo_height())

    lbl_user=Label(login_frm,bg=bgclr,font=('',15,''),fg='green',text='Welcome:Admin')
    lbl_user.place(x=10,y=100)

    back_btn=Button(login_frm,width=15,command=lambda:back(login_frm),font=('',12,'bold'),text='Back',bd=5)
    back_btn.place(relx=.85,y=100)

       
    browse_btn=Button(login_frm,width=20,height=h,command=lambda:browse(login_frm),font=('',12,'bold'),text='Browse',bd=5,bg='orange')
    browse_btn.place(x=400,y=200)
    
    detect_btn=Button(login_frm,width=20,height=h,command=lambda:Detect(dir_path,login_frm),font=('',12,'bold'),text='Detect',bd=5,bg='orange')
    detect_btn.place(x=700,y=200)


def use_video():
    print("wait for it")


def use_webcam():
    print("wait for it")


def back(frm):
    frm.destroy()
    welcomeAdmin()    
    

    #function for browse button
def browse(login_frm):
    global dir_path
    dir_path=filedialog.askopenfilename()
    im=Image.open(dir_path)
    im=im.resize((300,300),Image.ANTIALIAS)
    tkimage=ImageTk.PhotoImage(im)
    label=Label(login_frm,image=tkimage)
    label.image=tkimage
    label.place(x=300,y=400)
    
    

    
def Detect(dir_path,login_frm):


    
    img=cv2.imread(dir_path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = faceDetect.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        face=gray[y:y+h,x:x+w]
        face=cv2.resize(face,(90,90))
        print(np.argmax(loaded_model.predict(face.reshape(1,90,90,1)),axis=-1))
        a=np.argmax(loaded_model.predict(face.reshape(1,90,90,1)),axis=-1)
    
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        if(a==1):
                image=cv2.putText(img,"Male",(x-5,y-5),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
        else:
                image=cv2.putText(img,"Female",(x-5,y-5),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
        cv2.imwrite("facedetected.png",image)
        im=Image.open('facedetected.png')
        im=im.resize((300,300),Image.ANTIALIAS)
        tkimage=ImageTk.PhotoImage(im)
        label=Label(login_frm,image=tkimage)
        label.image=tkimage
        label.place(x=800,y=400)

      

def welcomeAdmin():
    width_of_btn=30
    h=4
    login_frm=Frame(root,bg=bgclr)
    login_frm.place(x=0,y=100,width=root.winfo_width(),height=root.winfo_height())

    lbl_user=Label(login_frm,bg=bgclr,font=('',15,''),fg='green',text='Welcome:Admin')
    lbl_user.place(x=10,y=100)

    logout_btn=Button(login_frm,width=15,command=lambda:logout(login_frm),font=('',12,'bold'),text='Logout',bd=5)
    logout_btn.place(relx=.85,y=100)

    use_image_btn=Button(login_frm,width=width_of_btn,height=h,command=lambda:use_image(login_frm),font=('',12,'bold'),text='Use Image',bd=5)
    use_image_btn.place(x=500,y=100)

    use_video_btn=Button(login_frm,width=width_of_btn,height=h,command=lambda:use_video(login_frm),font=('',12,'bold'),text='Use Video',bd=5)
    use_video_btn.place(x=500,y=200)

    web_cam_btn=Button(login_frm,width=width_of_btn,height=h,command=lambda:use_webcam(login_frm),font=('',12,'bold'),text='Use Webcam',bd=5)
    web_cam_btn.place(x=500,y=300)

#  login
def home():
    login_frm=Frame(root,bg=bgclr)
    login_frm.place(x=0,y=100,width=root.winfo_width(),height=root.winfo_height())

    lbl_user=Label(login_frm,bg=bgclr,font=('',20,''),fg='blue',text='Username')
    lbl_user.place(x=300,y=100)

    lbl_pass=Label(login_frm,bg=bgclr,font=('',20,''),fg='blue',text='Password')
    lbl_pass.place(x=300,y=150)

    ent_user=Entry(login_frm,bd=5,font=('',15,''))
    ent_user.focus()
    ent_user.place(x=490,y=105)

    ent_pass=Entry(login_frm,show='*',bd=5,font=('',15,'bold'))
    ent_pass.place(x=490,y=155)
    
    lgn_btn=Button(login_frm,text='login',command=lambda:login(login_frm,ent_user,ent_pass),bd=5,font=('',12,'bold'))
    lgn_btn.place(x=500,y=305)

    rst_btn=Button(login_frm,command=lambda:reset(ent_user,ent_pass),text='reset',bd=5,font=('',12,'bold'))
    rst_btn.place(x=590,y=305)
    
    
    

def reset(e1,e2,e3=None,e4=None):
    u=e1.get()
    p=e2.get()
    e1.delete(0,len(u))
    e2.delete(0,len(p))
    
    if(e3!=None and e4!=None):
        e=e3.get()
        m=e4.get()
        e3.delete(0,len(e))
        e4.delete(0,len(m))

def login(frm,e1,e2):
    u=e1.get()
    p=e2.get()
    if(len(u)==0 or len(p)==0):
        msg.showwarning('Validation Problem','Please fill all fields')
    else:
        if(u=='a' and p=='a'):
            msg.showinfo('Login','Welcome Admin')
            frm.destroy()
            welcomeAdmin()
        else:
            msg.showerror('Login Failed','Invalid username or password')        
            
def logout(frm):
    frm.destroy()
    home()










home()
root.mainloop()