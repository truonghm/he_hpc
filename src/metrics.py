import numpy as np

def ame(image1, image2):
	if image1.shape != image2.shape:
		raise ValueError('Image shapes do not match.')
	
	diff = np.abs(image1 - image2)
	ame = np.sum(diff) / image1.size

	return ame

def entropy(image):
	hist = np.histogram(image, bins=256, range=(0, 255))[0]
	hist = hist[hist != 0]
	entropy = -np.sum((hist / image.size) * np.log2(hist / image.size))

	return entropy