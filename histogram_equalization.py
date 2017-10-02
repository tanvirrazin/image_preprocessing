import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_images():
	images = []
	for ind in range(6):
		im = cv2.imread('./images/mra/{}.jpg'.format(ind+1))
		im = cv2.resize(im, (300, 300))
		images.append(im)

	return images

def show_images(original_images, global_hist_equ_images, adaptive_hist_equ_images):
	_, axarr = plt.subplots(3, 6)

	# Showing original images
	for ind, im in enumerate(original_images):
		axarr[0, ind].imshow(im)
		axarr[0, ind].set_title('Image {}'.format(ind+1))
		axarr[0, ind].axis('off')
		
	# Showing Global Histogram Equalization implemented images
	for ind, im in enumerate(global_hist_equ_images):
		axarr[1, ind].imshow(im,)
		axarr[1, ind].set_title('Image {}'.format(ind+1))
		axarr[1, ind].axis('off')
			
	# Showing Adaptive Histogram Equalization implemented images
	for ind, im in enumerate(adaptive_hist_equ_images):
		axarr[2, ind].imshow(im)
		axarr[2, ind].set_title('Image {}'.format(ind+1))
		axarr[2, ind].axis('off')

	plt.show('Histogram Equalization')


def normalize(images):
	normalized_images = []
	for ind, img in enumerate(images):
		normal_img = np.zeros((300, 300, 3))
		normal_img = cv2.normalize(img, normal_img, 0, 255, cv2.NORM_MINMAX)
		normalized_images.append(normal_img)

	return normalized_images


if __name__ == '__main__':
	original_images = read_images()

	normalized_images = normalize(original_images)


	# Converting to Single Channel gray-scale spectrum
	gray_images = []
	for im in normalized_images:
		gray_images.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))


	# Implementing Global histogram equalization
	global_equalized_images = []
	for im in gray_images:
		global_equalized_images.append(cv2.equalizeHist(im))


	# Implementing Adaptive histogram equalization
	adaptive_equalized_images = []
	for im in gray_images:
		clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
		cl1 = clahe.apply(im)
		adaptive_equalized_images.append(cl1)


	show_images(original_images, global_equalized_images, adaptive_equalized_images)


