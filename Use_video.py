import cv2

from tensorflow.keras.models import model_from_json
import numpy as np

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


def vid():

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    video_capture = cv2.VideoCapture("1.mp4")

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = faceDetect.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            faces=gray[y:y+h,x:x+w]
            faces=cv2.resize(faces,(90,90))
            # print(np.argmax(loaded_model.predict(faces.reshape(1,90,90,1)),axis=-1))
            # a=np.argmax(loaded_model.predict(faces.reshape(1,90,90,1)),axis=-1)
            a=1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if(a==1):
                cv2.putText(frame,"Male",(x-5,y-5),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
            else:
                cv2.putText(frame,"Female",(x-5,y-5),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)

        # Display the resulting frame
        cv2.putText(frame,"press q to close webcam",(10,10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
        cv2.imshow('Webcame facedetection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

vid()