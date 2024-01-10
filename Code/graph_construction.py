import cv2
import numpy as np
import random
import sys
import imageio
 
class GRAPH():

    def __init__(self, mark, binary, index):

        self.mark = mark[index]
        self.binary = binary[index]
              
    def rect_contains(self, rect, point):

        if point[0] < rect[0] :
            return False
        elif point[1] < rect[1] :
            return False
        elif point[0] > rect[2] :
            return False
        elif point[1] > rect[3] :
            return False
        return True
     
    def draw_point(self, img, p, color ):

        cv2.circle( img, (p[1], p[0]), 2, color, cv2.FILLED, 16, 0 )
     
    def draw_delaunay(self, img, subdiv, delaunay_color ):

        triangleList = subdiv.getTriangleList();
        size = img.shape
        r = (0, 0, size[0], size[1])

        slope_length = [[]]
        for i in range(self.mark.max()-1):
            slope_length.append([])

        for t in triangleList:
             
            pt1 = (int(t[0]), int(t[1]))
            pt2 = (int(t[2]), int(t[3]))
            pt3 = (int(t[4]), int(t[5]))
             
            if self.rect_contains(r, pt1) and self.rect_contains(r, pt2) and self.rect_contains(r, pt3):
                
                cv2.line(img, (pt1[1], pt1[0]), (pt2[1], pt2[0]), delaunay_color, 1, 16, 0)
                cv2.line(img, (pt2[1], pt2[0]), (pt3[1], pt3[0]), delaunay_color, 1, 16, 0)
                cv2.line(img, (pt3[1], pt3[0]), (pt1[1], pt1[0]), delaunay_color, 1, 16, 0)

                for p0 in [pt1, pt2, pt3]:
                    for p1 in [pt1, pt2, pt3]:
                        if p0 != p1:
                            temp = self.length_slope(p0, p1)
                            if temp not in slope_length[self.mark[p0]-1]:
                                slope_length[self.mark[p0]-1].append(temp)

        return slope_length

    def length_slope(self, p0, p1):

        if p1[1]-p0[1]:
            slope = (p1[0]-p0[0]) / (p1[1]-p0[1])
        else:
            slope = 1e10

        length = np.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)

        return length, slope

    def generate_points(self):

        centroids = []
        label = []
        max_label = self.mark.max()

        for i in range(1, max_label+1):
            img = self.mark.copy()
            img[img!=i] = 0
            if img.sum():
                contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
                m = cv2.moments(contours[0])

                if m['m00']:
                    label.append(i)
                    centroids.append(( int(round(m['m01']/m['m00'])),\
                                       int(round(m['m10']/m['m00'])) ))
                else:
                    label.append(i)
                    centroids.append(( 0,0 ))

        return centroids, label

    def run(self, animate = False):

        img_orig = self.binary.copy()

        size = img_orig.shape
        rect = (0, 0, size[0], size[1])
         
        subdiv = cv2.Subdiv2D(rect);
        
        points, label = self.generate_points()

        centroid = np.zeros((self.mark.max(), 2))
        for p, l in zip(points, label):
            centroid[l-1] = p

        outimg = []
        for p in points:
            subdiv.insert(p)
             
            if animate and False:
                img_copy = img_orig.copy()
                self.draw_delaunay( img_copy, subdiv, (255, 255, 255) )
                outimg.append(img_copy)
                cv2.imshow("win_delaunay", img_copy)
                cv2.waitKey(50)

        if len(outimg) != 0:
            imageio.mimsave('graph_contruction.gif', outimg, duration=0.3)

        slope_length = self.draw_delaunay( img_orig, subdiv, (255, 255, 255) )
     
        for p in points :
            self.draw_point(img_orig, p, (0,0,255))
        
        if animate and False:
            cv2.imshow('img_orig',img_orig)
            k = cv2.waitKey(0)
            if k == 27:        
                cv2.destroyAllWindows()
            elif k == ord('s'):
                cv2.imwrite('messigray.png',img)
                cv2.destroyAllWindows()

        return centroid, slope_length

def main():
    mark = [cv2.imread(sys.argv[1])]
    binary = [cv2.imread(sys.argv[2])]
    mark[0] = cv2.cvtColor(mark[0], cv2.COLOR_BGR2GRAY)
    binary[0] = cv2.cvtColor(binary[0], cv2.COLOR_BGR2GRAY)
    graph = GRAPH(mark, binary, 0)
    centroid, slope_length = graph.run(True)
    with open("centroid_slope_length.txt", "w+") as file:
        for i, p in enumerate(centroid):
            file.write(str(p[0])+" "+str(p[1])+"     "+str(slope_length[i])+"\n")

if __name__ == '__main__':
    main()