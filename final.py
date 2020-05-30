import matplotlib.pyplot as plt, argparse, numpy as np, math, sys, copy
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage import io
from collections import defaultdict
import numpy as np
import cv2


class SP:
    def __init__(self, img, step, nc):
        self.img = img
        self.height, self.width = img.shape[:2]
        self._convertToLAB()
        self.step = step
        self.nc = nc
        self.ns = step
        self.FLT_MAX = 1000000
        self.ITERATIONS = 10

    def _convertToLAB(self):

            import cv2
            self.labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
            self.im=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            #cv2.imshow("labima", self.im)
            self.sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)  # Find x and y gradients
            self.sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
            self.mag = np.sqrt(self.sobelx ** 2.0 + self.sobely ** 2.0)
            #self.arc = np.arctan(self.sobelx // self.sobely)
            #cv2.waitKey(0)
            #sp.generateSuperPixels()


    #cv2.imshow("super", lab[img])



    def generateSuperPixels(self):
        self._initData()
        indnp = np.mgrid[0:self.height, 0:self.width].swapaxes(0, 2).swapaxes(0, 1)
        g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        gm = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        ft = cv2.filter2D(gm, cv2.CV_8UC3, g_kernel)
        #print(indnp)
        for i in range(self.ITERATIONS):
            self.distances = self.FLT_MAX * np.ones(self.img.shape[:2])
            for j in range(self.centers.shape[0]):
                xlow, xhigh = int(self.centers[j][3] - self.step), int(self.centers[j][3] + self.step)
                ylow, yhigh = int(self.centers[j][4] - self.step), int(self.centers[j][4] + self.step)

                if xlow <= 0:
                    xlow = 0
                if xhigh > self.width:
                    xhigh = self.width
                if ylow <= 0:
                    ylow = 0
                if yhigh > self.height:
                    yhigh = self.height

                cropimg = self.labimg[ylow: yhigh, xlow: xhigh]
                colordiff= cropimg - self.labimg[int(self.centers[j][4]),int(self.centers[j][3])]
                colorDist = np.sqrt(np.sum(np.square(colordiff), axis=2))
                magimg=self.mag[ylow:yhigh,xlow:xhigh]
                mdif= abs(magimg - self.labimg[int(self.centers[j][4]),int(self.centers[j][3])])
                md=np.sqrt(np.sum((mdif), axis=2))
                htimg=ft[ylow:yhigh,xlow:xhigh]
                htdif= abs(htimg - self.labimg[int(self.centers[j][4]),int(self.centers[j][3])])
                hd=np.sqrt(np.sum((htdif), axis=2))
                #arcimg=self.arc[ylow:yhigh,xlow:xhigh]
                #adif = abs(arcimg - self.labimg[int(self.centers[j][4]), int(self.centers[j][3])])
               # ad = np.sqrt(np.sum((adif), axis=2))


                yy, xx = np.ogrid[ylow: yhigh, xlow: xhigh]
                pixdist = ((yy - self.centers[j][4]) ** 2 + (xx - self.centers[j][3]) ** 2) ** 0.5
                #dist = ((colorDist / self.nc) ** 2 + (pixdist / self.ns) ** 2) ** 0.5
                dist=np.sqrt(colorDist+pixdist+md+hd)

                distanceCrop = self.distances[ylow: yhigh, xlow: xhigh]
                idx = dist < distanceCrop
                #print(idx)

                distanceCrop[idx] = dist[idx]
                self.distances[ylow: yhigh, xlow: xhigh] = distanceCrop
                self.clusters[ylow: yhigh, xlow: xhigh][idx] = j

            for k in range(len(self.centers)):
                idx = (self.clusters == k)
                colornp = self.labimg[idx]
                distnp = indnp[idx]
                self.centers[k][0:3] = np.sum(colornp, axis=0)
                sumy, sumx = np.sum(distnp, axis=0)
                self.centers[k][3:] = sumx, sumy
                self.centers[k] /= np.sum(idx)
        #sp.createConnectivity()
    def _initData(self):
        self.clusters = -1 * np.ones(self.img.shape[:2])
        self.distances = self.FLT_MAX * np.ones(self.img.shape[:2])

        centers = []
        #print(centers)
        for i in range(self.step, self.width - self.step // 2, self.step):
            for j in range(self.step, self.height - self.step // 2, self.step):
                nc = self._findLocalMinimum(center=(i, j))
                #print(nc)
                color = self.labimg[nc[1], nc[0]]

                center = [color[0], color[1], color[2], nc[0], nc[1]]

                centers.append(center)
                #print(centers)
        self.center_counts = np.zeros(len(centers))
        self.centers = np.array(centers)
    def createConnectivity(self):
        label = 0
        adjlabel = 0
        lims = self.width * self.height // self.centers.shape[0]
        dx4 = [-1, 0, 1, 0]
        dy4 = [0, -1, 0, 1]
        new_clusters = -1 * np.ones(self.img.shape[:2]).astype(np.int64)
        elements = []
        for i in range(self.width):
            for j in range(self.height):
                if new_clusters[j, i] == -1:
                    elements = []
                    elements.append((j, i))
                    for dx, dy in zip(dx4, dy4):
                        x = elements[0][1] + dx
                        y = elements[0][0] + dy
                        if (x >= 0 and x < self.width and
                                y >= 0 and y < self.height and
                                new_clusters[y, x] >= 0):
                            adjlabel = new_clusters[y, x]
                count = 1
                c = 0
                while c < count:
                    for dx, dy in zip(dx4, dy4):
                        x = elements[c][1] + dx
                        y = elements[c][0] + dy

                        if (x >= 0 and x < self.width and y >= 0 and y < self.height):
                            if new_clusters[y, x] == -1 and self.clusters[j, i] == self.clusters[y, x]:
                                elements.append((y, x))
                                new_clusters[y, x] = label
                                count += 1
                    c += 1
                if (count <= lims >> 2):
                    for c in range(count):
                        new_clusters[elements[c]] = adjlabel
                    label -= 1
                label += 1
        self.new_clusters = new_clusters
        return new_clusters
        
        #sp.displayContours((252, 0, 0))
              

    def displayContours(self,color):
        dx8 = [-1, -1, 0, 1, 1, 1, 0, -1]
        dy8 = [0, -1, -1, -1, 0, 1, 1, 1]

        isTaken = np.zeros(self.img.shape[:2], np.bool)
        contours = []

        for i in range(self.width):
            for j in range(self.height):
                nr_p = 0
                for dx, dy in zip(dx8, dy8):
                    x = i + dx
                    y = j + dy
                    if x >= 0 and x < self.width and y >= 0 and y < self.height:
                        if isTaken[y, x] == False and self.clusters[j, i] != self.clusters[y, x]:
                            nr_p += 1

                if nr_p >= 2:
                    isTaken[j, i] = True
                    contours.append([j, i])
                
        for i in range(len(contours)):

            self.img[contours[i][0], contours[i][1]] = color
        return contours    
         
     #cv2.drawContours(int[[self.img[contours[i][0], contours[i][1]]]], 0, (0, 255, 0), 3)
       
   
    def _findLocalMinimum(self, center):
        min_grad = self.FLT_MAX
        loc_min = center
        #print(loc_min)
        for i in range(center[0] - 1, center[0] + 2):
            for j in range(center[1] - 1, center[1] + 2):
                c1 = self.labimg[j + 1, i]
                c2 = self.labimg[j, i + 1]
                c3 = self.labimg[j, i]
                if ((c1[0] - c3[0]) ** 2) ** 0.5 + ((c2[0] - c3[0]) ** 2) ** 0.5 < min_grad:
                    min_grad = abs(c1[0] - c3[0]) + abs(c2[0] - c3[0])
                    loc_min = [i, j]


        return loc_min
def initiateSegmentAttributes(segm_grid, image):
    
    def initialSegmAttr():
        return {'neighbors': set(), 'L': [], 'A': [], 'B': [], 'coord': set(),
                         'L_avg': 0.0, 'A_avg': 0.0, 'B_avg': 0.0}
    segm_dict = defaultdict(initialSegmAttr)
    print(segm_grid)

    for i in range(len(segm_grid)):
        for j in range(len(segm_grid[i])):
            if j != len(segm_grid[i]) - 1 and segm_grid[i][j] != segm_grid[i][j+1]:
                segm_dict[segm_grid[i][j]]['neighbors'].add(segm_grid[i][j+1])
                segm_dict[segm_grid[i][j+1]]['neighbors'].add(segm_grid[i][j])
            if i != len(segm_grid) - 1 and segm_grid[i][j] != segm_grid[i+1][j]:
                segm_dict[segm_grid[i][j]]['neighbors'].add(segm_grid[i+1][j])
                segm_dict[segm_grid[i+1][j]]['neighbors'].add(segm_grid[i][j])
            segm_dict[segm_grid[i][j]]['L'].append(image[i][j][0])
            segm_dict[segm_grid[i][j]]['A'].append(image[i][j][1])
            segm_dict[segm_grid[i][j]]['B'].append(image[i][j][2])
            segm_dict[segm_grid[i][j]]['coord'].add((i,j))
    return segm_dict

def getNearest(segm_dict):
    
    for k, v in segm_dict.items():
        v['L_avg'] = sum(v['L'])/len(v['L'])
        v['A_avg'] = sum(v['A'])/len(v['A'])
        v['B_avg'] = sum(v['B'])/len(v['B'])
    neighbor_pairs = set()
    nearest_neighbors = []
    shortest_dist = 100.0

    for k, v in segm_dict.items():
        for neighbor in v['neighbors']:
            neighbor_pair = tuple(sorted([k, neighbor]))
            if neighbor_pair not in neighbor_pairs and k != neighbor:
                neighbor_pairs.add(neighbor_pair)
                eucl_dist = float(math.sqrt((v['L_avg'] - segm_dict[neighbor]['L_avg']) ** 2 +
                                            (v['A_avg'] - segm_dict[neighbor]['A_avg']) ** 2 +
                                            (v['B_avg'] - segm_dict[neighbor]['B_avg']) ** 2))
                if eucl_dist < shortest_dist:
                    shortest_dist = eucl_dist
                    nearest_neighbors = neighbor_pair
    return nearest_neighbors, shortest_dist

def mergeSegments(segm_dict, nearest_neighbors):
   
    mergeto_dict = segm_dict[nearest_neighbors[0]]
    mergefrom_dict = copy.deepcopy(segm_dict[nearest_neighbors[1]])

    mergeto_dict['neighbors'] = mergeto_dict['neighbors'] | mergefrom_dict['neighbors']
    mergeto_dict['neighbors'].discard(nearest_neighbors[0])
    mergeto_dict['L'] += mergefrom_dict['L']
    mergeto_dict['A'] += mergefrom_dict['A']
    mergeto_dict['B'] += mergefrom_dict['B']
    mergeto_dict['coord'] = mergeto_dict['coord'] | mergefrom_dict['coord']

    for neighbor in mergefrom_dict['neighbors']:
        segm_dict[neighbor]['neighbors'].add(nearest_neighbors[0])
        segm_dict[neighbor]['neighbors'].discard(nearest_neighbors[1])

    del segm_dict[nearest_neighbors[1]]
    return segm_dict

def getSPHCsegments(segm_grid,image):
    
        print ("Initiating Segment Attributes...")
        segm_dict = initiateSegmentAttributes(segm_grid,image)
        shortest_dist = 0.0
        merge_count = 0

        print ("Merging Segments...")
        while shortest_dist < mx:
            nearest_neighbors, shortest_dist = getNearest(segm_dict)
            segm_dict = mergeSegments(segm_dict, nearest_neighbors)
            merge_count += 1
           # if merge_count % 20 == 0:
                #print( merge_count, "segments merged")

        print ( "segments merged - final")

        newSegmGrid = copy.deepcopy(segm_grid)
        for k, v in segm_dict.items():
            for coord in v['coord']:
                newSegmGrid[coord[0], coord[1]] = int(k)

        return newSegmGrid  


#imge = cv2.imread('fll.jpg')
imge = (io.imread('daisy.jpg'))


# imge = cv2.imread('egg.jpg')
plt.imshow(imge)
plt.show()
   


#def adjust_gamma(image, gamma=3.5):

   #invGamma = 1 / gamma
   #table = np.array([((i / 255.0) ** invGamma) * 255
      #for i in np.arange(0, 256)]).astype("uint8")

   #return cv2.LUT(image, table)
gamma = 1.5 # Gamma < 1 ~ Dark  ;  Gamma > 1 ~ Bright

 

#img = ((imge/255) ** (1/gamma))
m=np.array(255*(imge/255)**1,dtype='uint8')
img=cv2.medianBlur(m,5)
#cv2.imshow('original',original)
plt.imshow(img)
plt.show()


#gamma = 3.5
#img = adjust_gamma(original, gamma=gamma)
imlab=cv2.cvtColor(imge,cv2.COLOR_BGR2LAB)


#cv2.putText(img, "g={}".format(gamma), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
#cv2.imshow("gammam image 1", img)

#cv2.imshow("input",img)
nr_superpixels = int(400)
nc = int(6)
step = int((img.shape[0] * img.shape[1] / nr_superpixels) ** 0.5)
mx=np.std(img)

sp = SP(img, step, nc)
sp.__init__(img,step,nc)
#print(r)
sp.generateSuperPixels()
r=sp.createConnectivity()
#cv2.imshow("cluster", slic.clusters.astype(np.uint8))
#slic.displayContours()

sp.displayContours((252, 0, 0))

#sp.getSPHCsegments(contours,img)  

#cv2.imshow("superpixels", img)
SPHCsegm_grid =getSPHCsegments(r,imlab) 

plt.imshow(img)
plt.show()
fig = plt.figure("%d Segments Merged" % 400)
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(imge, SPHCsegm_grid))
plt.axis("off")
plt.show()

#cv2.imshow("color",color)
#cv2.waitKey(0)
#cv2.imshow("SLICimg.jpg", slic.img)
#plt.imshow(SPHCsegm_grid)
#plt.show()

#cv2.waitKey(0)
