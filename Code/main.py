import cv2
import sys
import os
import numpy as np
import imageio
from adaptivethresh import ADPTIVETHRESH as ADPT
from gvf import GVF
from matplotlib import pyplot as plt
from watershed import WATERSHED as WS
from graph_construction import GRAPH
from matching import FEAVECTOR as FEA
from matching import SIMPLE_MATCH as MAT
from scipy import spatial as sp
from scipy import ndimage
from scipy.spatial import distance

def normalize(image):
    img = image.copy().astype(np.float32)
    img -= np.mean(img)
    img /= np.linalg.norm(img)
    img = np.clip(img, 0, 255)
    img *= (1./float(img.max()))
    return (img*255).astype(np.uint8)

path = os.path.join("resource/training/01")
images = []
enhance_images = []
for r,d,f in os.walk(path):
    for files in f:
        if files[-3:].lower()=='tif':
            temp = cv2.imread(os.path.join(r,files))
            gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY) 
            images.append(gray.copy())
            enhance_images.append(normalize(gray.copy()))

print("Total number of image is ", len(images))
print("The shape of image is ", images[0].shape, type(images[0][0,0]))

def write_image(image, title, index, imgformat='.tiff'):
    if index < 10:
            name = '0'+str(index)
    else:
        name = str(index)
    cv2.imwrite(title+name+imgformat, image)

def main():

    # # Binarization
    adaptive = ADPT(enhance_images)
    threh = adaptive.applythresh(50)
    for i, img in enumerate(threh):
        threh[i] = img * 255

    # Nuclei center detection
    gvf = GVF(enhance_images, threh)
    dismap = gvf.distancemap()
    newimg = gvf.new_image(0.4, dismap) 
    out = []
    pair = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    for i,img in enumerate(dismap):
        neighborhood_size = 20
        data_max = ndimage.filters.maximum_filter(img, neighborhood_size)
        data_max[data_max==0] = 255
        pair.append((img == data_max).astype(np.uint8))
        y, x = np.where(pair[i]>0)
        points = list(zip(y[:], x[:]))
        dmap = distance.cdist(points, points, 'euclidean')
        y, x = np.where(dmap<20)
        ps = zip(y[:], x[:])
        for p in ps:
            if p[0] != p[1]:
                pair[i][points[min(p[0], p[1])]] = 0
        dilation = cv2.dilate((pair[i]*255).astype(np.uint8),kernel,iterations = 1)
        out.append(dilation)

    # watershed
    print("test watershed")
    ws = WS(newimg, pair) 
    wsimage, binarymark, marks = ws.watershed_compute()

    os.chdir("test1")
    for i in range(len(images)):
        write_image(threh[i].astype(np.uint8), 'threh', i)
        write_image(images[i].astype(np.uint8), 'images', i)
        write_image(enhance_images[i].astype(np.uint8), 'enhance_images', i)
        write_image(newimg[i].astype(np.uint8), 'newimg', i)
        write_image(dilation, "seed_point", i)
    os.chdir(os.pardir)

    os.chdir("test")
    for i in range(len(images)):
        write_image(marks[i].astype(np.uint8), 'mark_train_01/mark', i)
        write_image(binarymark[i].astype(np.uint8), 'binarymark_train_01/binmark', i)
    os.chdir(os.pardir)

    # centroid = []
    # slope_length = []
    # # Build Delaunay Triangulation
    # print("test triangulation")
    # for i in range(len(images)):
    #     graph = GRAPH(marks, binarymark, i)
    #     tempcentroid, tempslope_length = graph.run(True)
    #     centroid.append(tempcentroid)
    #     slope_length.append(tempslope_length)

    # # Build the Dissimilarity measure vector
    # print("test dissimilarity")
    # vector = []
    # for i in range(len(images)):
    #     print("  feature vector: image ", i)
    #     v = FEA()
    #     v.set_centroid(centroid[i])
    #     v.set_spatial(slope_length[i])
    #     v.set_shape(images[i], marks[i])
    #     v.set_histogram()
    #     v.add_label()
    #     v.add_id(marks[i].max(), i)
    #     vector.append(v.generate_vector())
        
    #     print("num of nuclei: ", len(vector[i]))

    # # Feature matching
    # mask = []
    # for i in range(len(images)-1):
    #     print("  Feature matching: image ", i)
    #     m = MAT(i,i+1,[images[i], images[i+1]], vector)
    #     mask.append(m.generate_mask(marks[i], i)) 
    #     m.find_match(0.3)
    #     mask = m.match_missing(mask, max_frame=2, max_distance=20)
    #     vector[i+1] = m.mitosis_refine()
    #     m.new_id()  
    #     vector[i+1] = m.return_vectors()

    # os.chdir("test")
    # for i in range(len(images)):
    #     write_image(marks[i].astype(np.uint8), 'mark_train_01/marks', i)
    #     write_image(binarymark[i].astype(np.uint8), 'binarymark_train_01/binarymark', i)

if __name__ == '__main__':
    main()