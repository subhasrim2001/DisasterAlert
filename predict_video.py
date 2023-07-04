# USAGE: as function in flask frontend
# import the necessary packages
from keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
from twilio.rest import Client
def prediction(filename, o_filename):
    camid = 'IRNSS-1I'
    location = 'Geosynchronous / 55°E, 29° inclined orbit'
    
    # Example: 'example_clips/hurricane.mp4', 'output/hurricane_output.mp4'
    args = {'model':'model/activity_gpu.model' , 'label_bin':'model/lb.pickle', 'input':filename, 'output':o_filename, 'size':10}

    print("[INFO] loading model and label binarizer...")
    model = load_model(args["model"])
    lb = pickle.loads(open(args["label_bin"], "rb").read())

    mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
    Q = deque(maxlen=args["size"])
    
    vs = cv2.VideoCapture(args["input"])
    writer = None
    (W, H) = (None, None)
    ok = 'Normal'
    fi_label = []
    framecount = 0
    while True:
    	(grabbed, frame) = vs.read()
    	if not grabbed:
    		break
    	if W is None or H is None:
    		(H, W) = frame.shape[:2]
    	framecount = framecount+1
    	output = frame.copy()
    	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    	frame = cv2.resize(frame, (224, 224)).astype("float32")
    	frame -= mean
    	preds = model.predict(np.expand_dims(frame, axis=0))[0]
    	prediction = preds.argmax(axis=0)
    	Q.append(preds)
    	results = np.array(Q).mean(axis=0)
		# debugging print statements
    	print('Results = ', results)
    	maxprob = np.max(results)
    	print('Maximun Probability = ', maxprob)
    	i = np.argmax(results)
    	print('Maximun Probability = ', maxprob)
    	print(lb.get_params())
    	print(lb.classes_)
    	label = lb.classes_[i]
    	print(label)
    	rest = 1 - maxprob
    	diff = (maxprob) - (rest)
    	print('Difference of prob ', diff)
    	th = 100
    	if diff > .80:
    		th = diff
    	if (preds[prediction]) < th:
    		text = "Alert : {} - {:.2f}%".format((ok), 100 - (maxprob * 100))
    		cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
    	else:
    		fi_label = np.append(fi_label, label)
    		text = "Alert : {} - {:.2f}%".format((label), maxprob * 100)
    		cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
    		prelabel = label    


    	# Commented writer (to disk) for security - can uncomment if necessary
    	# if writer is None:
    	# 	# initialize our video writer
    	# 	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    	# 	writer = cv2.VideoWriter(args["output"], fourcc, 30,
    	# 		(W, H), True)
    	# writer.write(output)

    	# show the output image
    	cv2.imshow("Output", output)
    	key = cv2.waitKey(1) & 0xFF
    	# if the `q` key is pressed, break from the loop
    	if key == ord("q"):
    		break
    print('Frame count', framecount)
    print('Count label', fi_label)
    # release the file pointers - if writer is uncommented, uncomment this as well
    # print("[INFO] cleaning up...")
    # writer.release()
    # vs.release()