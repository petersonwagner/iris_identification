#!/usr/bin/python

import cv2, os
import cv2.cv as cv
import cv as cv1
import numpy as np
from PIL import Image
import math
import pywt
np.set_printoptions(threshold=np.nan)


def get_iris_radius (img, circles):
	img = cv2.equalizeHist(img)
	circles = np.uint8(np.around(circles))


	circles[0,0,2] += 30
	circle_img1 = np.zeros(img.shape, dtype = 'uint8')
	for i in circles[0,:]:
		cv2.circle(circle_img1,(i[0],i[1]),i[2],255,1)	# draw the outer circle

	rect_img = np.zeros(img.shape, dtype = 'uint8')
	cv2.rectangle(rect_img, (0, int(circles[0,0,1] + circles[0,0,2]*0.7)), (1000, int(circles[0,0,1] - circles[0,0,2]*0.7)), 255, -1)
	
	circle_img2 = cv2.bitwise_and(circle_img1, rect_img)
	circle_img3 = cv2.bitwise_and(circle_img2, img)

	circle_points = np.extract (circle_img3, img)
	mean_circle_prev = np.mean (circle_points)

	#print '\n\n'
	for x in xrange(1,40):
		circles[0,0,2] += 1
		circle_img1 = np.zeros(img.shape, dtype='uint8')
		for i in circles[0,:]:
			cv2.circle(circle_img1,(i[0],i[1]),i[2],255,1)	# draw the outer circle

		rect_img = np.zeros(img.shape, dtype = 'uint8')
		cv2.rectangle(rect_img, (0, int(circles[0,0,1] + circles[0,0,2]*0.7)), (1000, int(circles[0,0,1] - circles[0,0,2]*0.7)), 255, -1)

		circle_img2 = cv2.bitwise_and(circle_img1, rect_img)
		circle_img3 = cv2.bitwise_and(circle_img2, img)		

		circle_points = np.extract (circle_img3, img)
		mean_circle = np.mean (circle_points)

		diff = mean_circle - mean_circle_prev
		if (diff > 6):
			#print diff
			return circles[0,0,2]

		mean_circle_prev = mean_circle

	return circles[0,0,2]


def theta_transform (theta, M):
    return ((2*math.pi*theta) / (M-1))

def rho_transform(rho, rho_max, rho_min, R):
    return ((rho * (rho_max - rho_min) / (R-1) ) + rho_min)

def bilinear_interpolation(img, x, y):
    #print img[int(y),int(x)]
    x1 = int(math.floor(x))
    x2 = int(math.ceil(x))
    y1 = int(math.floor(y))
    y2 = int(math.ceil(y))

    if y1 >= 280:
    	y1 = 279
    if y2 >= 280:
    	y2 = 279
    if x1 >= 320:
    	x1 = 319
    if x2 >= 320:
    	x2 = 319
    if x >= 320:
    	x = 319
    if y >= 280:
    	y = 279

    if x1 == x2:
        f_xy1 = img[y1,int(x)]
        f_xy2 = img[y2,int(x)]
    else:
        f_xy1 = (x2 - x)/(x2 - x1) * img[y1,x1] + (x - x1)/(x2 - x1) * img[y1,x2]
        f_xy2 = (x2 - x)/(x2 - x1) * img[y2,x1] + (x - x1)/(x2 - x1) * img[y2,x2]

    if y1 == y2:
        return f_xy1
    else:
        f_xy = (y2 - y)/(y2 - y1) * f_xy1 + (y - y1)/(y2 - y1) * f_xy2

    return f_xy


def normalize_img (img, pupil_radius, iris_radius, x_center, y_center):
	normalized = np.zeros((300,70), dtype='uint8')
	for i in xrange(0,300):
	    for j in xrange(0,70):
	        new_theta = theta_transform (theta=i, M=300)
	        new_rho = rho_transform (rho=j, rho_min=pupil_radius, rho_max=iris_radius, R=70)

	        x = new_rho * math.cos(new_theta) + x_center
	        y = new_rho * math.sin(new_theta) + y_center

	        normalized[i,j] = int(round(bilinear_interpolation (img, y, x)))

	normalized = np.rot90(normalized, 1)
	return normalized


class Subject:
	def __init__(self, path):
		self.path = path
		self.L_path = path + '/L'
		self.R_path = path + '/R'
		self.L_images = []
		self.R_images = []
		self.iris_code = []


	def read_images(self):
		#reading subject's left eye
		self.L_paths = [os.path.join(self.L_path, f) for f in os.listdir(self.L_path)]
		self.L_paths.sort()

		for L_path in self.L_paths:
			new_image = cv2.imread(L_path, 0)
			self.L_images.append(new_image)


		#reading subject's right eye
		self.R_paths = [os.path.join(self.R_path, f) for f in os.listdir(self.R_path)]
		self.R_paths.sort()

	def train(self):
		for path in [item for sublist in [self.L_paths,self.R_paths] for item in sublist]:
			print path
			original_image = cv2.imread(path, 0)
			self.R_images.append(original_image)
			blurred_image = cv2.medianBlur(original_image, 5) #####

			th, im_th = cv2.threshold(blurred_image, 80, 255, cv2.THRESH_BINARY_INV)

			kernel = np.ones((3,1),np.uint8)
			erosion1 = cv2.erode(im_th, kernel, iterations = 1)

			kernel = np.ones((1,3),np.uint8)
			erosion2 = cv2.erode(erosion1, kernel, iterations = 1)

			mask = np.zeros((282, 322), np.uint8)
			im_floodfill = erosion2.copy()
			cv2.floodFill(im_floodfill, mask, (300,200), 255);

			im_floodfill_inv = cv2.bitwise_not(im_floodfill)

			pupil_mask = erosion2 | im_floodfill_inv
			pupil_mask = cv2.bitwise_not(pupil_mask)

			preprocessed = cv2.bitwise_and(pupil_mask, original_image)

			#edges = cv2.Canny(pupil_mask,100,200)
			circles = cv2.HoughCircles(image=cv2.GaussianBlur(pupil_mask, (5,5), 0),method=cv.CV_HOUGH_GRADIENT,dp=2,minDist=220,param1=500,param2=50,minRadius=20,maxRadius=70)

			if circles is None:
				break

			pupil_radius = circles[0,0,2]
			iris_radius = get_iris_radius (original_image, circles)

			normalized = normalize_img(original_image, pupil_radius, iris_radius, circles[0,0,1], circles[0,0,0])
			cv2.imshow('norm', normalized)
			cv2.waitKey(0)


			wav = pywt.wavedec2(normalized, 'haar', level=4)
			cA = wav[0]
			(cH, cV, cD) = wav[1]

			feature_template = (cH + cV + cD)/3
			self.iris_code = (feature_template >= 0) * 1  #binarizing image with threshold_value 0
			


			print feature_template
			print self.iris_code





			



path = "./CASIA-IrisV4-Interval"

subj_paths = [os.path.join(path, f) for f in os.listdir(path)]
subj_paths.sort()

subjects = []

for subj_path in subj_paths:
	new_subj = Subject(subj_path)
	subjects.append(new_subj)
	
for subject in subjects:
	subject.read_images()
	subject.train()




#obs:
#applying hough circles on pupil_mask instead of canny edges