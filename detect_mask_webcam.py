# USAGE
# python3 detect_mask_webcam.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from twilio.rest import Client
from datetime import datetime
import RPi.GPIO as GPIO
from time import sleep



ena = 26
enb = 2
in1 = 19
in2 = 13
in3 = 6
in4 = 5
button = 11
led1 = 16
led2 = 20
led3 = 21
trig1 = 22
echo1 = 27
trig2 = 9
echo2 = 10


GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
GPIO.setup(ena, GPIO.OUT)
GPIO.setup(enb, GPIO.OUT)
GPIO.setup(led1, GPIO.OUT)
GPIO.setup(led2, GPIO.OUT)
GPIO.setup(led3, GPIO.OUT)
GPIO.setup(trig1, GPIO.OUT)
GPIO.setup(echo1, GPIO.IN)
GPIO.setup(trig2, GPIO.OUT)
GPIO.setup(echo2, GPIO.IN)
GPIO.setup(button, GPIO.IN, pull_up_down = GPIO.PUD_UP)
GPIO.output(in1, GPIO.LOW)
GPIO.output(in2, GPIO.LOW)
GPIO.output(in3, GPIO.LOW)
GPIO.output(in4, GPIO.LOW)
GPIO.output(led1, GPIO.LOW)
GPIO.output(led2, GPIO.LOW)
GPIO.output(led3, GPIO.LOW)
p = GPIO.PWM(ena, 1000)
q = GPIO.PWM(enb, 1000)

p.start(100)
q.start(100)

max_passenger = 30
total = 0
c_mask = 0
c_nomask = 0
door_status = 0
c_noface = 0
entry_detected = 0
exit_detected = 0
waiting = 0
#sms_sent = 0

def door_open():
	GPIO.output(in3, GPIO.HIGH)
	GPIO.output(in4, GPIO.LOW)
	GPIO.output(in1, GPIO.LOW)
	GPIO.output(in2, GPIO.HIGH)
	GPIO.output(led1, GPIO.HIGH)
	print("open")
	sleep(4)
	GPIO.output(in1, GPIO.LOW)
	GPIO.output(in2, GPIO.LOW)
	GPIO.output(in3, GPIO.LOW)
	GPIO.output(in4, GPIO.LOW)
	GPIO.output(led1, GPIO.LOW)

def door_open_exit():
	GPIO.output(in3, GPIO.HIGH)
	GPIO.output(in4, GPIO.LOW)
	GPIO.output(in1, GPIO.LOW)
	GPIO.output(in2, GPIO.HIGH)
	GPIO.output(led3, GPIO.HIGH)
	print("open")
	sleep(4)
	GPIO.output(in1, GPIO.LOW)
	GPIO.output(in2, GPIO.LOW)
	GPIO.output(in3, GPIO.LOW)
	GPIO.output(in4, GPIO.LOW)
	GPIO.output(led3, GPIO.LOW)
    
def door_closed():
	GPIO.output(in1, GPIO.HIGH)
	GPIO.output(in2, GPIO.LOW)
	GPIO.output(in3, GPIO.LOW)
	GPIO.output(in4, GPIO.HIGH)
	print("close")
	sleep(4)
	GPIO.output(in1, GPIO.LOW)
	GPIO.output(in2, GPIO.LOW)
	GPIO.output(in3, GPIO.LOW)
	GPIO.output(in4, GPIO.LOW)
    
def ultrasonic1():
    GPIO.output(trig1, False)
    sleep(0.2)
        
    GPIO.output(trig1, True)
    sleep(0.00001)
    GPIO.output(trig1, False)
        
        
    while GPIO.input(echo1) == 0:
        start_time1 = time.time()
    
    while GPIO.input(echo1) == 1:
        stop_time1 = time.time()
    
    duration1 = stop_time1 - start_time1
    distance1 = duration1 * 17150
    return distance1

def ultrasonic2():
    GPIO.output(trig2, False)
    sleep(0.2)
    GPIO.output(trig2, True)
    sleep(0.00001)
    GPIO.output(trig2, False)
        
    while GPIO.input(echo2) == 0:
        start_time2 = time.time()
        
    while GPIO.input(echo2) == 1:
        stop_time2 = time.time()
    
    duration2 = stop_time2 - start_time2
    distance2 = duration2 * 17150
    return distance2
    
def green_led():
	GPIO.output(led1, GPIO.HIGH)
	sleep(2)
	GPIO.output(led1, GPIO.LOW)
	
def red_led():
	GPIO.output(led2, GPIO.HIGH)
	sleep(1)
	GPIO.output(led2, GPIO.LOW)

def out_led():
	GPIO.output(led3, GPIO.HIGH)
	sleep(2)
	GPIO.output(led3, GPIO.LOW)
	
def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="facemask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
	c_noface = c_noface + 1
	waiting = 0
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	path = '/home/pi/Desktop/project/Mask_detection/bg_pc.jpg'
	image = cv2.imread(path)
	image = imutils.resize(image, width=1000)
	label_1 = "No. of Passengers:"
	label_2 = "{} / {}".format(total, max_passenger)
	label_3 = "Please wear your mask properly."
	label_4 = "OVERLOADING!"
	color_1 = (255,255,255)
	color_2 = (0,255,127)
	color_3 = (0,140,255)
	color_4 = (0,0,255)
	cv2.putText(image, label_1, (10,70), cv2.FONT_HERSHEY_DUPLEX, 2, color_1, 3)
	
	frame = vs.read()
	frame = imutils.resize(frame, width=350)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		# determine the class label and color we'll use to draw
		# the bounding box and text
		if mask > withoutMask:
			label = "Mask"
			color = (0, 255, 0)
			c_mask = c_mask + 1
			c_nomask = 0

		else:
			label = "No Mask"
			color = (0, 0, 255)
			c_nomask = c_nomask + 1
			c_mask = 0
			#c_noface = 0
		
		## include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
	
	door_stat = door_status % 2
	dist1 = ultrasonic1()
	dist2 = ultrasonic2()
	
	button_state = GPIO.input(button)
	if button_state == 0:
		c_noface = 0
		if door_stat == 0:
			door_open()
			door_status = door_status + 1
		else:
			door_closed()
			door_status = door_status + 1

			
	if dist1 <= 70:
		entry_detected = 1
	else:
		entry_detected = 0
		
	if dist2 <= 30:
		exit_detected = 1
		waiting = 1
	else:
		exit_detected = 0
		waiting = 0
	
	if c_mask == 3 and entry_detected == 1 and waiting == 0:
		total = total + 1
		c_mask = 0
		c_noface = 0
		if door_stat == 0:
			door_open()
			door_status = door_status + 1
		else:
			green_led()
	
	if c_mask >= 3 and entry_detected == 0:
		c_mask = 0
	
	if exit_detected == 1:
		c_noface = 0
		if total > 0:
			total = total - 1
		if door_stat == 0:
			door_open_exit()
			door_status = door_status + 1
		else:
			out_led()
		
	if c_noface >= 30:
		if door_stat == 1:
			door_closed()
			door_status = door_status + 1
			
	if c_noface == 1:
		c_nomask = 0
			
	cv2.putText(image, label_2, (200,300), cv2.FONT_HERSHEY_TRIPLEX, 5, color_2, 5)

	if c_nomask > 0 and entry_detected == 1:
		cv2.putText(image, label_3, (10,400), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color_3, 2)
		red_led()

	if total > max_passenger:
		cv2.putText(image, label_4, (50,500), cv2.FONT_HERSHEY_TRIPLEX, 3, color_4, 5)
	
		if sms_sent == 0:
			now = datetime.now()
			today = now.strftime("%B %d, %Y %H:%M:%S")
			client = Client("AC8cd32a54f401b3817e313acfc25393a2","ed8fd4a547dd5e857ea49826336fe537")
			client.messages.create(body = "\n {} \n BUS OVERLOADING! \n Bus Driver: John Rey Mozo \n Bus Conductor: Allan Robert Buzon \n Plate No: 12345 \n Route: Tubigon - Tagbilaran (Vice Versa)".format(today),
			from_ = '+14196706120',
			to = '+639656081549')
			sms_sent = 1
	
	if total <= max_passenger:
		sms_sent = 0
	
        

	# show the output frame
	cam_win = "Camera Frame"
	cv2.namedWindow(cam_win)
	cv2.moveWindow(cam_win, 1000,450)
	cv2.imshow(cam_win, frame)
	
	win_pc = "Passeger Counter"
	cv2.namedWindow(win_pc)
	cv2.moveWindow(win_pc, 1,1)
	cv2.imshow(win_pc, image)
	
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
