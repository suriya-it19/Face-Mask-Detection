import os
import time
import urllib.request
import cv2
import imutils
import numpy as np
import pandas as pd
import tensorflow
from imutils.video import FPS, VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from twilio.rest import Client
from flask import Flask, render_template, Response
from camera import Camera

thresold = 0.5
Sadam,Santhosh,Deepak=0,0,0
data = pd.read_excel('students.xlsx')

app = Flask(__name__, template_folder='templates')

def detect_and_predict_mask(frame, faceNet, maskNet,face_model):

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()

	faces = []
	locs = []
	preds = []
	preds1 = []

	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		if confidence > thresold:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
		preds1 = face_model.predict(faces, batch_size=32)

	return (locs, preds, preds1)


def message_alert(name):
    data = pd.read_excel('students.xlsx')
    count = 0
    # Your Account Sid and Auth Token from twilio.com/console
    # and set the environment variables. See http://twil.io/secure
    account_sid = 'AC0622bdd42a99e44fe4ecd051df9dd0c6' 
    auth_token = '0d205db9464d0cc627c32eab573546a2' 
    client = Client(account_sid, auth_token)
    for names in data['Name'].values:
        if names == name:
            message = client.messages \
                            .create(
                                body="{} You have been warned for violating the rules.Please wear Mask for your as well as others Safety".format(str(data.iloc[count,0])),
                                media_url=['https://image.shutterstock.com/image-vector/safety-first-symbol-260nw-102323386.jpg'],
                                
                                from_='+918870966785',
                                to='+91{}'.format(data.iloc[count,1])
                            )

            print(message.sid)
        count += 1


print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(['face-mask-detector/face_detector', "deploy.prototxt"])
weightsPath = os.path.sep.join(['face-mask-detector/face_detector',"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading face mask detector model...")
maskNet = load_model('Model\Final_model_50.h5')
face_model = load_model('F:/Projects/Mask-detection/face-recognition/normal_main_model_150.h5')

print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = cv2.VideoCapture(0)
url='http://192.168.1.2:8080/shot.jpg'
time.sleep(2.0)

def gen_frames(Sadam,Santhosh,Deepak):  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = vs.read()  # read the camera frame
        frame = imutils.resize(frame, width=700)


        (locs, preds, preds1) = detect_and_predict_mask(frame, faceNet, maskNet, face_model)

        for (box, pred) in zip(locs, preds):

            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label1 = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label1 == "Mask" else (0, 0, 255)
            face_color = (0, 255, 0)

            label = "{}: {:.2f}%".format(label1, max(mask, withoutMask) * 100)
            if label1 == 'No Mask':
                for i in preds1:
                    face_pred = np.argmax(i)        
                    if (i[face_pred] > 0.5):
                        if (face_pred == 0):
                            face_label = 'Deepak'
                            Deepak = Deepak + 1
                            Sadam,Santhosh=0,0
                            if Deepak > 20:
                                print('Warning Deepak')
                            #    message_alert('Deepak')
                                Deepak = 0
                        if (face_pred == 1):
                            face_label  = 'Sadam'
                            Sadam += 1
                            Santhosh,Deepak=0,0
                            if Sadam > 20:
                                print('Warning Sadam')
                            #    message_alert('Sadam')
                                Sadam = 0
                        if (face_pred == 2):
                            face_label = 'Santhosh'
                            Santhosh += 1
                            Sadam,Deepak=0,0
                            if Santhosh > 20:
                                print('Warning Santhosh')
                            #    message_alert('Santhosh')
                                Santhosh = 0
                frame_label = 'Sadam: {0}, Santhosh: {1}, Deepak: {2}'.format(Sadam,Santhosh,Deepak)

                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.putText(frame, face_label, (startX, startY - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, face_color, 2)
                cv2.putText(frame, frame_label, (startX, startY - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, face_color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                #Sadam,Santhosh,Deepak=0,0,0

            else:
                Sadam,Santhosh,Deepak=0,0,0
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

        
def gen_frames1(Sadam,Santhosh,Deepak):  # generate frame by frame from camera
    while True:        
        imgPath = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgPath.read()),dtype=np.uint8)
        img2 = cv2.imdecode(imgNp,-1)
        img2 = imutils.resize(img2,width=700)
        #Sadam,Santhosh,Deepak=0,0,0

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds, preds1) = detect_and_predict_mask(img2, faceNet, maskNet,face_model)

        for (box, pred) in zip(locs, preds):

            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label1 = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label1 == "Mask" else (0, 0, 255)
            face_color = (0, 255, 0)

            label = "{}: {:.2f}%".format(label1, max(mask, withoutMask) * 100)
            if label1 == 'No Mask':
                for i in preds1:
                    face_pred = np.argmax(i)        
                    if (i[face_pred] > 0.5):
                        if (face_pred == 0):
                            face_label = 'Deepak'
                            Deepak = Deepak + 1
                            Sadam,Santhosh=0,0
                            if Deepak > 20:
                                print('Warning Deepak')
                            #    message_alert('Deepak')
                                Deepak = 0
                        if (face_pred == 1):
                            face_label  = 'Sadam'
                            Sadam += 1
                            Santhosh,Deepak=0,0
                            if Sadam > 20:
                                print('Warning Sadam')
                            #    message_alert('Sadam')
                                Sadam = 0
                        if (face_pred == 2):
                            face_label = 'Santhosh'
                            Santhosh += 1
                            Sadam,Deepak=0,0
                            if Santhosh > 20:
                                print('Warning Santhosh')
                            #    message_alert('Santhosh')
                                Santhosh = 0
                frame_label = 'Sadam: {0}, Santhosh: {1}, Deepak: {2}'.format(Sadam,Santhosh,Deepak)

                cv2.putText(img2, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.putText(img2, face_label, (startX, startY - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, face_color, 2)
                cv2.putText(img2, frame_label, (startX, startY - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, face_color, 2)
                cv2.rectangle(img2, (startX, startY), (endX, endY), color, 2)
                #Sadam,Santhosh,Deepak=0,0,0

            else:
                Sadam,Santhosh,Deepak=0,0,0
                cv2.putText(img2, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(img2, (startX, startY), (endX, endY), color, 2)


        ret, buffer = cv2.imencode('.jpg', img2)
        img2 = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + img2 + b'\r\n')  # concat frame one by one and show result


@app.route('/webcam')
def cam_video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(Sadam,Santhosh,Deepak), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mobile')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames1(Sadam,Santhosh,Deepak), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data', methods=("POST", "GET"))
def html_table():

    return render_template('base.html',  tables=[data.to_html(classes='data')], titles=data.columns.values)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run()



