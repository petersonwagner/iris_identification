#!/usr/bin/python

import cv2, os
import cv2.cv as cv
import cv as cv1
import numpy as np
from PIL import Image
import math
import pywt
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)


def get_iris_radius (img, circles):
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
		self.img_list = []


	def read_images(self):
		#reading subject's left eye
		self.L_paths = [os.path.join(self.path + '/L', f) for f in os.listdir(self.path + '/L')]
		self.L_paths.sort()

		#reading subject's right eye
		self.R_paths = [os.path.join(self.path + '/R', f) for f in os.listdir(self.path + '/R')]
		self.R_paths.sort()


	def get_iris_features(self):
		for path in [item for sublist in [self.L_paths,self.R_paths] for item in sublist]:
			
			#
			print 'computing features: ' + path
			#

			original_image = cv2.imread(path, 0)
			
			mask_path = './CASIA-IrisV4-Interval-mask/' + path[30:-4] + '.tiff'
			mask_image = cv2.imread(mask_path, 0)

			mask = np.zeros((282, 322), np.uint8)
			cv2.floodFill(mask_image, mask, (0,0), 255);

			preprocessed = cv2.bitwise_and(mask_image, original_image)

			edges = cv2.Canny(mask_image,100,200)
			#circles = cv2.HoughCircles(image=cv2.GaussianBlur(mask_image, (5,5), 0),method=cv.CV_HOUGH_GRADIENT,dp=2,minDist=220,param1=500,param2=50,minRadius=20,maxRadius=70)
			circles = cv2.HoughCircles(image=cv2.GaussianBlur(preprocessed, (5,5), 0),method=cv.CV_HOUGH_GRADIENT,dp=2,minDist=220,param1=500,param2=50,minRadius=20,maxRadius=70)

			if circles is None:
				break

			pupil_radius = circles[0,0,2]
			equalized_img = cv2.equalizeHist(original_image)
			iris_radius = get_iris_radius (equalized_img, circles)

			normalized = normalize_img(original_image, pupil_radius, iris_radius, circles[0,0,1], circles[0,0,0])
			
			wav = pywt.wavedec2(normalized, 'haar', level=4)
			cA = wav[0]
			(cH, cV, cD) = wav[1]

			feature_template = (cH*3 + cV + cD)/5
			

			iris_code = (feature_template >= 0)  #binarizing image with threshold_value 0
			label = path[30:-4]
			
			self.img_list.append([label, iris_code.flatten()])

			#cv2.imshow('norm', normalized)
			#cv2.waitKey(0)


def hamming_distance(array1, array2):
	n = array1.shape[0]
	xor_op = np.logical_xor(array1, array2).sum()
	return xor_op / float(n)


path = "./CASIA-IrisV4-Interval"

subj_paths = [os.path.join(path, f) for f in os.listdir(path)]
subj_paths.sort()

subjects = []

for subj_path in subj_paths:
	new_subj = Subject(subj_path)
	subjects.append(new_subj)
	
for subject in subjects:
	subject.read_images()
	subject.get_iris_features()

print 'computing metrics...'

img_list_all = [img_list for subj in subjects for img_list in subj.img_list]

image_num = len(img_list_all)
distance_matrix = np.zeros((image_num, image_num), dtype=object)
values = np.zeros((image_num, image_num), dtype=float)


for i in xrange(0,image_num):
	for j in xrange(0,image_num):
		a = img_list_all[i]
		b = img_list_all[j]
		distance_matrix[i,j] = [hamming_distance(a[1], b[1]), a[0][:-2] == b[0][:-2], a[0] + ',' + b[0]]
		values[i,j] = hamming_distance(a[1], b[1])

upper_indices = np.triu_indices(image_num)
distance_matrix = distance_matrix[upper_indices]
distance_matrix = np.sort(distance_matrix)

values = np.unique(values)
values = np.append(values, 1.0)


positives_total = float(0)
negatives_total = float(0)

for dist in distance_matrix:
	if dist[1] == True:
		positives_total += 1
	else:
		negatives_total += 1


false_positives = []
false_negatives = []

for threshold in values:
	fp = 0
	fn = 0

	for dist in distance_matrix:
		if dist[0] <= threshold and dist[1] == False:
			fp += 1
		elif dist[0] > threshold and dist[1] == True:
			fn += 1

	fp = fp / float(negatives_total)
	fn = fn / float(positives_total)
	false_positives.append(fp)
	false_negatives.append(fn)

false_positives = np.array(false_positives)
false_negatives = np.array(false_negatives)

differences = np.absolute(false_positives - false_negatives)
index_EER = np.argmin(differences)

print 'false_positives:'
print false_positives
print 'false_negatives:'
print false_negatives
print 'EER: {},{}'.format(false_positives[index_EER], false_negatives[index_EER])


plt.axis([-0.1, 1.1, -0.1, 1.1])
plt.plot(false_positives, false_negatives, linestyle='--')
plt.plot(false_positives[index_EER], false_negatives[index_EER], 'o', color='red')
plt.show()