import numpy as np
import argparse
import cv2
import time
import os


# Parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to input video file")
args = vars(ap.parse_args())

# open a pointer to the video stream and start the FPS timer
stream = cv2.VideoCapture(args["video"])

# Resolution
W = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
resolution_str = "Res: "+str(W)+"x"+str(H)

# FPS
frameCounter = 0
timeBegin   = time.time();
average_fps = 0


while(True):
	# Capture frame-by-frame
	ok, frame = stream.read() # Read method both GET and DECODE frame (bad practice)

	if not ok:
		break

	frameCounter += 1

	if (frameCounter % 10 == 0):
		elapsedTime = time.time() - timeBegin
		average_fps = round(frameCounter / elapsedTime, 2)
		
		os.system('clear')
		print(resolution_str)
		print("Frame: "+str(frameCounter))
		print("Average FPS: "+str(average_fps))

	# show the frame and update the FPS counter
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(delay=1) ## 1 milliseconds
	if key==27:    # Esc key to stop   ####  & 0xFF == ord('q'):
		break


# When everything done, release the capture
stream.release()
cv2.destroyAllWindows()