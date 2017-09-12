import numpy as np
import cv2
from matplotlib import pyplot as plt


img = cv2.imread('./images/brain_aneurysm.jpeg')

# As OpenCV uses BGR pattern
# and Matplotlib uses RGB pattern
b, g, r = cv2.split(img)
rgb_img = cv2.merge([r, g, b])

result_images = []

def denoise_image(h):
	dst = cv2.fastNlMeansDenoising(img, None, h, 7, 21)
	b, g, r = cv2.split(dst)
	return cv2.merge([r, g, b])

# Reduced noise with 8 different h values
for h in [3, 5, 7, 10, 12, 14, 16, 20]:
	result_images.append(denoise_image(h))


# Plotting original image
plt.subplot(331).set_title('Original image'), plt.imshow(rgb_img)

# Plotting result images after denoising
for i in range(1, 9):
	plt.subplot(331+i).set_title('Result {}'.format(i))
	plt.imshow(result_images[i-1])

plt.show()
