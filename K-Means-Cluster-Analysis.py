## Simulation Using Python: K-Means Cluster Analysis

##This project is for a cell phone company to establish its network by putting its towers in a particular region it has acquired. 
##The location of these towers can be found by clustering (where data points are residence locations) 
##so that all users receive optimum signal strength. Used numpy, matplotlib, K-means clustering.


```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 5 2018
@author: Hazel
This is to perform a K-Means cluster analysis, but the focus will be on the
 centers of the clusters.
Topic: A cell phone company needs to establish its network by putting its
 towers in a particular region it has acquired. The location of these towers
 can be found by clustering (where data points are residence locations) so that
 all users receive optimum signal strength.
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import math
# Two data sets to represent different size communities.
# The first data set will represent a small city with 100 homes uniformly
 distributed within a 10X10 square mile region.
city1 = np.random.uniform(low=-5.0, high=5.0, size=(100,2))
# The second data set will represent a larger city with 1000 homes within a
 20X20 square mile region,
city2 = np.random.uniform(low=-10.0, high=10.0, size=(1000,2))
# with more homes distributed inside the 10X10 square mile region around the
city center.
cityOverall = np.concatenate((city1, city2), axis = 0)
'''
    This function chooses k random elements from the data array to be the
     initial centers of the k clusters.
PARAMS:
    data_array: the two-dimensional array of data
    k: the number of clusters
'''
def initClusterCenters(data_array,k):
    centers = np.zeros((k,2))   # create k cluster centers
    for i in range(k):      # for each cluster, choose a random integer in the
        ind = random.randint(0,len(data_array)-1)   # range of possible indices
        cpt = (data_array[ind,0],data_array[ind,1])   # set the center point
        while cpt in centers:       # make sure that point isn't already used
as a center
            ind = random.randint(0,len(data_array)-1)
            cpt = (data_array[ind,0],data_array[ind,1])
        centers[i] = (data_array[ind,0],data_array[ind,1])  # set the point in
         the centers array
    return centers
'''
    This function computes the Euclidean distance between two points
PARAMS:

    pt1: a tuple/list/array of size 1x2
    pt2: a tuple/list/array of size 1x2
'''
def distance(pt1,pt2):
    d =  math.sqrt((pt1[0]-pt2[0])**2 + (pt2[1]-pt2[1])**2)
    return d
'''
    Determine which cluster an individual pt should be in
PARAMS:
    centers: the list of cluster centers
    pt: the point to determine the cluster for
'''
def labelForPt(centers,pt):
    minD = distance(centers[0], pt)
    ind = 0
    for i in range(len(centers)):
        c = centers[i]
        # calculate the distance from pt to c
        newD = distance(c,pt)
        # if that distance is less than the minimum,
        if newD < minD:
        #   replace the minimum distance and the corresponding index
minD = newD
            ind = i
    return ind
'''
    This function determines which clusters all of the points in the data array
     belong to.
PARAMS:
    data_array: the two-dimensional array of data
    centers: the array of points representing the cluster centers
'''
def determineLabels(data_array, centers):
    # replace the following assignment of labels with something more useful
    # you may do this in one line, or several lines
    # your code should involve the following call, presuming i goes
    # over all indices of data_array
    #        labelForPt(centers,(data_array[i,0],data_array[i,1]))
    labels = np.zeros(len(data_array))
    for i in range(len(data_array)):
        labels[i] = labelForPt(centers,(data_array[i,0],data_array[i,1]))
    return labels
'''
    In the data array, the particular cluster number is existed in labels.
    The data array with the cluster information returns.
'''
def extractCluster(dataArray,labels,clusterNum):
    indices = np.where(labels == clusterNum)

    cluster = dataArray[indices]
    return cluster
'''
    This function recalculates the centers of the clusters, based on the mean
     values of the elements in the clusters.
PARAMS:
    centers: the current cluster centers
    clusterList: the list of clusters (sets of pts belonging to each cluster)
'''
def recalculateCenters(centers,clusterList):
    for i in range(len(clusterList)):
        c = clusterList[i]
        # Calculate the mean of the elements in the first column of c
        meanFirstCol = np.mean(c[:,0])
        # calculate the mean of the elements in the second column of c
        meanSecondCol = np.mean(c[:,1])
        centers[i] = (meanFirstCol,meanSecondCol) # should be (mean of first
         column,mean of second column)
    return centers
"""
    This function takes a list of points in a cluster, along with the center
     point of that cluster as parameters,
    and returns the maximum and minimum distances between any point in that
     list and its center.
"""
def maxMinDistance(data_array, pt):
    minD = distance(data_array[0], pt)
    maxD = distance(data_array[0], pt)
    for i in range(len(data_array)):
        c = data_array[i]
        # calculate the distance from pt to c
        newD = distance(c,pt)
        # if that distance is less than the minimum,
        #   replace the minimum distance and the corresponding index.
        if newD < minD:
            minD = newD
        if newD > maxD:
            maxD = newD
    return maxD, minD
''' This is the main function that runs the K-Mean algorithm '''
def main():
    #  Ask user how many clusters
    k = int(input("How many clusters would you like to see? Or, how many towers
     do you want to build? "))

#
iteration = int(input("How many iterations would you like to see? "))
cityOverallCenters = initClusterCenters(cityOverall,k)
print("initial centers for city overall:",cityOverallCenters)
# iteration
for n in range(iteration):
    # Determine which cluster the points belong to
    labels = determineLabels(cityOverall,cityOverallCenters)
    plt.figure()
    clusterList = []
    colorList=['green','cyan','blue', 'yellow','magenta','pink']
    for i in range(k):
         # Get the actual sets of points
        c = extractCluster(cityOverall,labels,i)
        clusterList.append(c)
        # Plot the clusters
        plt.axis([-iteration-5,iteration+5, -iteration-5, iteration+5])
        plt.scatter(c[:,0],c[:,1],color=colorList[i])    #
         np.random.rand(3,1))
        maxMin = maxMinDistance(c, cityOverallCenters[i])
        for j in range(len(cityOverallCenters)):
            plt.plot(cityOverallCenters[j,0],cityOverallCenters[j,1],'k^')
        plt.pause(0.5)
    # recalculate the centers
    cityOverallCenters = recalculateCenters(cityOverallCenters,clusterList)
print("new centers:", cityOverallCenters)
print("maximum and minimum distance:", maxMin)
```
