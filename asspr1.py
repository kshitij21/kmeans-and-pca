__author__ = 'kshitij'
__author__ = 'kshitij'


from PIL import Image
from numpy import *
from PIL import Image
from pylab import *
from numpy import mean,cov,cumsum,dot,linalg,size,flipud
from numpy import *
import matplotlib.pyplot as plt
from pandas import *
def pca(X):
  # Principal Component Analysis
  # input: X, matrix with training data as flattened arrays in rows
  # return: projection matrix (with important dimensions first),
  # variance and mean

  #get dimensions


  num_data,dim = X.shape
  print dim


  #center data
  mean_X = X.mean(axis=0)
  for i in range(num_data):
      X[i] -= mean_X



      #print 'PCA - compact trick used'
      #print 1
      M = dot(X,X.T) #covariance matrix
      #print DataFrame(M)
      e,EV = linalg.eigh(M) #eigenvalues and eigenvectors
      #print EV
      #E=EV[::-2]
      #print E
      tmp = dot(X.T,EV).T #this is the compact trick
      #print tmp
      V = tmp[::-1] #reverse since last eigenvectors are the ones we want
      #print V
      F=dot(V.T,EV).T

      S = (e)[::-1] #reverse since eigenvalues are in increasing order
      #print S


  #return the projection matrix, the variance and the mean
  return V,S,mean_X,F


#X=  random.rand(3,2)*2
#X=random.uniform(low=0.75, high=1.5, size=(2,3) )

#X = imread('/home/kshitij/Desktop/tweety.jpeg') # load an image
#print X
#print pca(X)


from PIL import Image
import numpy
import pylab
import os

imlist=['/home/kshitij/Downloads/gwb_cropped/images/18.jpg','/home/kshitij/Downloads/gwb_cropped/images/19.jpg','/home/kshitij/Downloads/gwb_cropped/images/20.jpg']
from PIL import Image
import numpy
import pylab

im = numpy.array(Image.open(imlist[0])) #open one image to get the size
m,n = im.shape[0:2] #get the size of the images
imnbr = len(imlist) #get the number of images

#create matrix to store all flattened images
immatrix = numpy.array([numpy.array(Image.open(imlist[i])).flatten() for i in range(imnbr)],'f')

#perform PCA
V,S,immean,F = pca(immatrix)

#mean image and first mode of variation
immean = immean.reshape(m,n)
mode = V[0].reshape(m,n)
#print mode

#show the images
'''
pylab.figure()
pylab.gray()
pylab.imshow(immean)
'''

pylab.figure()
pylab.gray()
pylab.imshow(mode)

pylab.show()

