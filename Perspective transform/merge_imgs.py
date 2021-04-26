import cv2
import numpy as np
from pathlib import Path

KEYPOINT_LEN       = 10
KEYPOINT_COLOR     = (36,12,255) # red (BGR)
KEYPOINT_THICKNESS = 1

class KeyPointLabeler():

	def __init__(self, img_path: str, resize_pct=.25):

		self.img_path      = img_path
		self.img           = cv2.imread(self.img_path)
		if resize_pct: self.img = self.resize(self.img, resize_pct)
		self.img_keypoints = self.img.copy()
		self.keypoints     = []

		cv2.namedWindow(img_path)
		cv2.setMouseCallback(img_path, self.mouse_events)

	def resize(self, img, resize_pct=.5):
		width  = int(img.shape[1] * resize_pct )
		height = int(img.shape[0] * resize_pct)
		return cv2.resize(img, (width, height))

	def show(self):
		cv2.imshow(self.img_path, self.img_keypoints)


	def mouse_events(self, event, x, y, flags, parameters):
		
		# MOVE MOUSE -> Show fancy cursor
		if event == cv2.EVENT_MOUSEMOVE:
			self.img_keypoints = self.img.copy()
			cv2.line(self.img_keypoints, (x-KEYPOINT_LEN,y-KEYPOINT_LEN), (x+KEYPOINT_LEN,y+KEYPOINT_LEN), KEYPOINT_COLOR, thickness=KEYPOINT_THICKNESS)
			cv2.line(self.img_keypoints, (x-KEYPOINT_LEN,y+KEYPOINT_LEN), (x+KEYPOINT_LEN,y-KEYPOINT_LEN), KEYPOINT_COLOR, thickness=KEYPOINT_THICKNESS)

		# MOUSE RELEASE LEFT CLICK -> Save width and height
		elif event == cv2.EVENT_LBUTTONUP:
			cv2.line(self.img, (x-KEYPOINT_LEN,y-KEYPOINT_LEN), (x+KEYPOINT_LEN,y+KEYPOINT_LEN), KEYPOINT_COLOR, thickness=KEYPOINT_THICKNESS)
			cv2.line(self.img, (x-KEYPOINT_LEN,y+KEYPOINT_LEN), (x+KEYPOINT_LEN,y-KEYPOINT_LEN), KEYPOINT_COLOR, thickness=KEYPOINT_THICKNESS)
			self.keypoints.append([x,y])


path     = Path("imgs")
labeler1 = KeyPointLabeler(img_path=str(path/"calle1.jpg"))
labeler2 = KeyPointLabeler(img_path=str(path/"calle2.jpg"))
labeler3 = KeyPointLabeler(img_path=str(path/"calle3.jpg"))

while True:
	labeler1.show()
	labeler2.show()
	#labeler3.show()

	# Close program with keyboard "ESCAPE"
	key = cv2.waitKey(1)
	if key == 27:
		cv2.destroyAllWindows()
		exit(1)

	if key == ord("d"):

		pts1 = np.array(labeler1.keypoints, np.float32)
		pts2 = np.array(labeler2.keypoints, np.float32)
		pts3 = np.array(labeler3.keypoints, np.float32)

		print()
		print("img1", pts1)
		print("img2", pts2)
		print("img3", pts3)

		if (len(pts1)==4 or len(pts2)==4 ):
			M = cv2.getPerspectiveTransform(pts2, pts1) #"Calle2 ---> calle2 transformation"

			img2_transformed = cv2.warpPerspective(labeler2.img, M, (5000,4000))
			cv2.imshow("transofrmed", img2_transformed)
		else:
			print("Not 4 points")

