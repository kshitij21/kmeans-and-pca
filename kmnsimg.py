__author__ = 'kshitij'

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import scipy

from pylab import imread,imshow,figure,show,subplot
from numpy import reshape,uint8,flipud
from scipy.cluster.vq import kmeans,vq
'''
img = imread('/home/kshitij/Desktop/tweety.jpeg')
#imshow(flipud(img))
# reshaping the pixels matrix

pixel = reshape(img,(img.shape[0]*img.shape[1],3))
a=pixel.astype(np.float)
print (a)

# performing the clustering

centroids,_ = kmeans(a,5) # six colors will be found
print centroids

# quantization
qnt,_ = vq(pixel,centroids)

# reshaping the result of the quantization
centers_idx = reshape(qnt,(img.shape[0],img.shape[1]))
clustered = centroids[centers_idx]

figure(1)
subplot(211)
imshow(flipud(img))
subplot(212)
imshow(flipud(clustered))

show()

plt.hist(centroids,bins=10)

show()
'''
from numpy import *
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cbook import _string_to_bool

import Image
import pandas as pd
img = Image.open("/home/kshitij/Desktop/tweety.jpeg")
img = array(img, dtype=float64) / 255
w, h, d = original_shape = tuple(img.shape)
image_array = reshape(img, (w * h, d))
#print image_array


# init centroids with random samples
def ini_Centroids(data, k):
    numSamples, dim = data.shape
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = data[index, :]
    return centroids

# calculate Euclidean distance
def euc_Distnce(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))

# k-means cluster
def kmeans(data, k):
    numSamples = data.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clster_Assment = mat(zeros((numSamples, 2)))
    cluster_Changed = True

    ## step 1: init centroids
    centroids = ini_Centroids(data, k)

    while cluster_Changed:
        cluster_Changed = False
        ## for each sample
        for i in xrange(numSamples):
            minDist  = 1000.0
            minIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = euc_Distnce(centroids[j, :], data[i, :])
                if distance < minDist:
                    minDist  = distance
                    minIndex = j

            ## step 3: update its cluster
            if clster_Assment[i, 0] != minIndex:
                clusterChanged = True
                clster_Assment[i, :] = minIndex, minDist**2

        ## step 4: update centroids
        for j in range(k):
            pointsInCluster = data[nonzero(clster_Assment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis = 0)
    global labels
    labels = squeeze(asarray((clster_Assment[:,[0]]), dtype=int))
    #print labels
    #print clusterAssment
    
    return centroids

#print kmeans(image_array,2)
def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    #print codebook
    return image

df = pd.DataFrame(image_array)
#print df

plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized Image')
plt.imshow(recreate_image(kmeans(image_array, 6), labels, w, h))

label = pd.DataFrame(labels)


X = list(range(0,w))*h
x = pd.DataFrame(X)
Y=[]
count1 = 0
for j in range(h):

	for i in range(w):
		Y.append(count1)
	count1 = count1 + 1
y = pd.DataFrame(Y)

f=[df,label,x,y]
data = pd.concat(f, axis=1)
data.columns = ['r','g','b','labels','x','y']
#print data.tail()

x_hist=[]
y_hist=[]
r_hist=[]
g_hist=[]
b_hist=[]
for k in range(len(data)):
	if data['labels'][k] == 4 :
		x_hist.append(data['x'][k])
		y_hist.append(data['y'][k])
		r_hist.append(data['r'][k])
		g_hist.append(data['g'][k])
		b_hist.append(data['b'][k])
#print len(data)
#print x_hist
'''
plt.figure(2)
plt.title("P(x/C1)")
plt.hist(x_hist)
plt.figure(3)
plt.title("P(y/C1)")
plt.hist(y_hist)
plt.figure(4)
plt.title("P(r/C1)")
plt.hist(r_hist)
plt.figure(5)
plt.title("P(g/C1)")
plt.hist(g_hist)
plt.figure(6)
plt.title("P(b/C1)")
plt.hist(b_hist)

plt.show()
'''

hist, xedges, yedges = np.histogram2d(x_hist, y_hist, bins=4)
elements = (len(xedges) - 1) * (len(yedges) - 1)

xpos, ypos = np.meshgrid(xedges[:-1]+0.25, yedges[:-1]+0.25)

xpos = xpos.flatten()           # x-coordinates of the bars

ypos = ypos.flatten()           # y-coordinates of the bars
zpos = np.zeros(elements)       # zero-array
dx =  np.ones_like(zpos)   # length of the bars along the x-axis
dy = dx.copy()                  # length of the bars along the y-axis
dz = hist.flatten()

fig = plt.figure()
ax = Axes3D(fig)

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b')



plt.show()