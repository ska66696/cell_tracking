import cv2
import numpy as np
import sys
from numpy import unique

class WATERSHED():

	def __init__(self, images, markers):
		self.images = images
		self.markers = markers

	def watershed_compute(self):

		result = []
		outmark = []
		outbinary = []

		for i in range(len(self.images)):
			imgcolor = np.zeros((self.images[i].shape[0], self.images[i].shape[1], 3), np.uint8)
			for c in range(3): 
				imgcolor[:,:,c] = self.images[i]

			if len(self.markers[i].shape) == 3:
				self.markers[i] = cv2.cvtColor(self.markers[i],cv2.COLOR_BGR2GRAY)
			_, mark = cv2.connectedComponents(self.markers[i])

			mark = cv2.watershed(imgcolor, mark)

			u, counts = unique(mark, return_counts=True)
			counter = dict(zip(u, counts))
			for i in counter:
				if counter[i] > 1200:
					mark[mark==i] = 0

			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
			mark = cv2.morphologyEx(mark.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
			_, mark = cv2.connectedComponents(mark.astype(np.uint8))

			temp = cv2.cvtColor(imgcolor,cv2.COLOR_BGR2GRAY)
			result.append(temp)
			outmark.append(mark.astype(np.uint8))

			binary = mark.copy()
			binary[mark>0] = 255
			outbinary.append(binary.astype(np.uint8))

		return result, outbinary, outmark

def main():

	images = []
	image = cv2.imread(sys.argv[1])
	images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
	markers = []
	marker = cv2.imread(sys.argv[2])
	markers.append(cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY))

	ws = WS(newimg, imgpair) 
	wsimage, binarymark, mark = ws.watershed_compute()

	cv2.imwrite('binarymark.tif', (np.clip(binarymark, 0, 255)).astype(np.uint8))

if __name__ == '__main__':
    main()
	