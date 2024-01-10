import cv2
import sys
import numpy as np

class ADPTIVETHRESH():

    def __init__(self, images):
        self.images = []
        for img in images:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.images.append(img.copy())

    def applythresh(self, threshold = 129):

        out = []
        markers = []
        binarymark = []

        for img in self.images:
            img = cv2.GaussianBlur(img,(5,5),0).astype(np.uint8)
            _, thresh = cv2.threshold(img,threshold,1,cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            out.append(thresh)

        return out

def main():

    img = [cv2.imread(sys.argv[1])]
    adaptive = ADPTIVETHRESH(img)
    thresh, markers = adaptive.applythresh(10)
    cv2.imwrite("adaptive.tiff", thresh[0]*255)

if __name__ == '__main__':
    main()