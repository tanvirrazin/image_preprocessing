import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)


sigma = 0.155
original = imread('./images/brain_aneurysm.jpeg')

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 5), sharex=True,
                       sharey=True, subplot_kw={'adjustable': 'box-forced'})

plt.gray()

sigma_est = estimate_sigma(original, multichannel=True, average_sigmas=True)

ax[0, 0].imshow(original)
ax[0, 0].axis('off')
ax[0, 0].set_title('Original')

ax[0, 1].imshow(denoise_tv_chambolle(original, weight=0.08, multichannel=True))
ax[0, 1].axis('off')
ax[0, 1].set_title('Total Variation')

ax[1, 0].imshow(denoise_bilateral(original, sigma_color=0.05, sigma_spatial=15, multichannel=True))
ax[1, 0].axis('off')
ax[1, 0].set_title('Bilateral')

ax[1, 1].imshow(denoise_wavelet(original, multichannel=True))
ax[1, 1].axis('off')
ax[1, 1].set_title('Wavelet denoising')

fig.tight_layout()

plt.show()
