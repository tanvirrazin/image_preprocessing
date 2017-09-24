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

def show_images(images):
	for ind, im in enumerate(images):
		plt.subplot(231+ind).set_title('Image {}'.format(ind+1))
		plt.axis('off')
		plt.imshow(im)
	plt.show()


def normalize(images):
	normalized_images = []
	for ind, img in enumerate(images):
		normal_img = np.zeros((300, 300, 3))
		normal_img = cv2.normalize(img, normal_img, 0, 255, cv2.NORM_MINMAX)
		normalized_images.append(normal_img)

	return normalized_images


if __name__ == '__main__':
	images = read_images()

	normalized_images = normalize(images)

	# Implemented PCA for dimensionality reduction
	flattened_images = np.array([img.reshape(300*300*3) for img in normalized_images])
	m, V = cv2.PCACompute(flattened_images, np.mean(flattened_images, axis=0).reshape(1, -1))

	pca_images = []
	for i in range(6):
		pca_images.append(V[i].reshape(300, 300, 3))


	# Converting to Single Channel gray-scale spectrum
	gray_images = []
	for im in pca_images:
		gray_images.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))

	# Implementing histogram equalization
	equalized_images = []
	for im in gray_images:
		equalized_images.append(cv2.equalizeHist(im))

	show_images(gray_images)


