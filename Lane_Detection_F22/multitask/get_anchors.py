'''
This file combines the K-Means clustering implementation and the get_anchors implementation into one
file for efficiency purposes.
Author: William Stevens
Progress:
- 11/6: Merged kmeans.py and get_anchors() into one file.
'''

import pandas as pd
import numpy as np

import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt

from utils.utils import xyxy_to_xywh


'''
get_anchors function:
- Created by William Stevens on 11/5
- This function iterates over an entire portion of the BDD100k dataset and retrieves the bounding box information
for every object in every image. It then inputs an array of points containing bounding box widths and
heights into a Kmeans class object with k=5 for the algorithm to determine the 5 most common bounding
box sizes and aspect ratios in the entire dataset. These will serve as the anchor boxes.
'''
def get_anchors(root='/Users/calixkim/VIP27920/bdd100k/', val=True):
    '''
    Read the validation or training json file using Pandas
    '''
    if val:
        json = pd.read_json(root + 'labels/det_20/det_val.json')
        num_images = 10000
    else:
        json = pd.read_json(root + 'labels/det_20/det_train.json')
        num_images = 70000

    '''
    For the images in the validation or training portion of the BDD100k dataset, parse the json file to
    retrieve the bounding box coordinates (top left pixel and top right pixel).
    '''
    wh = []
    for i in range(num_images):
        '''
        Parse the json to retrieve just the current image data and single out the 'labels' section as
        this holds the bounding box coordinate information.
        '''
        target = json.iloc[i]
        annotations = target['labels']

        '''
        Evaluate the current image data's corresponding image in the dataset and obtain its width and
        height. These values will be used if we decide to normalize the values.
        '''
        if val:
            img_path = root + 'images/100k/val/'
        else:
            img_path = root + 'images/100k/train/'

        img = read_image(img_path + target['name']) 
        img = img.type(torch.float32)
        _, height, width = img.shape

        '''
        For each object in the current image data, parse the labels to obtain the 'box2d' section
        which holds the four coordinates of interest. Pass these coordinates into our xyxy_to_xywh
        function to convert them into the necessary format. Then append the width and height values
        to the list of widths and heights.
        '''
        for obj in annotations:
            bbox = list(obj['box2d'].values())
            bbox = xyxy_to_xywh(bbox, width, height, norm=False) 
            wh.append([bbox[2], bbox[3]])

    '''
    Make data into numpy array and use the zip function over the iterable to obtain just the widths and
    just the heights, for plotting purposes. After all points have been plotted, create a Kmeans object 
    and call its fit() function to obtain the anchor box dimensions.
    '''
    data = np.array(wh)
    kmeans = Kmeans(data, 9)
    return kmeans.fit()



'''
K-Means Clustering Implementation by William Stevens
Written to be used in the determination of anchor box priors for YOLO v3 as outlined
in the YOLO9000 and YOLO v3 papers.
(https://arxiv.org/pdf/1612.08242.pdf, https://arxiv.org/pdf/1804.02767.pdf)

Progress:
- 10/30/22: Wrote up implementation of Kmeans class and its necessary functions
- 11/1/22: Added explanatory comments throughout code
- 11/5/22: Ran sanity tests and fixed small bugs causing errors
'''
class Kmeans:
    def __init__(self, data, num_clusters):
        '''
        Initialize Kmeans object
        - Data parameter corresponds to an array of points corresponding to pixel coordinates
        - Num_cluster parameter corresponds to the number of clusters (k) to identify
        '''
        self.data = data
        self.k = num_clusters

    def get_centroid_labels(self, centroids):
        '''
        This function will return a numpy array that is parrallel to the data array. The indices will match,
        but the contents will differ since this array will contain the nearest centroid for each point. For
        example, data_labels[0] will contain the coordinates of the centroid point that data[0] is closest to.
        '''
        data_labels = []
        '''
        For each point in the data array, find the closest centroid
        '''
        for i in range(self.data.shape[0]):
            '''
            This will retrieve the ith multidimensional point. Using self.data[i, :] instead of self.data[i]
            will ensure that this function operates on pixels of multiple dimensions, but it will probably
            not be useful for the way we are using it. Might change this in the future.
            '''
            point = self.data[i, :]

            '''
            Initialize distance as infinity to begin with so that we can effectively minimize centroid distances
            '''
            distance = np.Inf
            '''
            For each centroid in the current centroids list, calculate the distance from the current point.
            Update distance and closest centroid variables if necessary.
            '''
            for i, centroid in enumerate(centroids):
                ith_centroid_dist = np.sqrt((point[0] - centroids[i][0])**2 + (point[1] - centroids[i][1])**2)
                '''
                If the ith centroid's distance is smaller than the current closest centroid's, update the distance
                and the closest_centroid variable
                '''
                if distance > ith_centroid_dist:
                    distance = ith_centroid_dist
                    closest_centroid = centroid
            data_labels.append(closest_centroid)
        
        '''
        Return a numpy array that is parrallel to the data array, where the contents contain the nearest
        centroid for each point. 
        '''
        return np.array(data_labels)

    def compute_centroids(self, prev_centroids, labels):
        '''
        This function will compute new centroids by iterating through each centroid's cluster, as defined per
        the data labels, and calculate means for that cluster's x-values and y-values. The new centroid's coordinates
        for each cluster will simply become those means.
        '''
        new_centroids = []

        '''
        For each centroid in the previously computed array of centroids...
        '''
        for centroid in prev_centroids:
            points = []
            '''
            Iterate through all the datapoints and retrieve those which have been assigned to the current
            centroid, as defined per the data labels.
            '''
            for i in range(self.data.shape[0]):
                '''
                If the current point has been assigned to the current centroid, add it to the previously
                defined list of points for later use in mean calculations.
                '''
                if (np.array_equal(labels[i], centroid)):
                    point = self.data[i]
                    points.append(point)

            x_values = []
            y_values = []
            '''
            For each point in the list of points that have been assigned to the current centroid, create
            separate lists of their x-values and y-values for later use in mean calculations.
            '''
            for p in points:
                x_values.append(p[0])
                y_values.append(p[1])

            '''
            Redefine the current centroid's coordinates as the mean x-coordinate and mean y-coordinate of
            all the points that were assigned to it. Add this new centroid to a list.
            '''
            new_centroid = [sum(x_values) / len(x_values), sum(y_values) / len(y_values)]
            new_centroids.append(new_centroid)

        '''
        Return an array of newly computed centroid coordinates.
        '''
        return np.array(new_centroids)
    
                
    def fit(self):
        '''
        This function will fit centroids over the datapoints by executing multiple iterations of data
        label assignments and centroid computations.
        '''

        '''
        First initialize k random centroids from the list of datapoints.
        '''
        centroids = self.data[np.random.choice(len(self.data), self.k, replace=False)]

        '''
        This centroids_changed flag will become False when an iteration of the K-means algorithm
        results in no change in computed centroids, meaning the algorithm has converged upon its
        centroid coordinates and the while loop is terminated.
        '''
        centroids_changed = True
        while (centroids_changed):
            '''
            For each iteration of the algorithm, obtain data labels for each point's closest centroid, and
            then use these data labels to compute new centroid coordinates.
            '''
            labels = self.get_centroid_labels(centroids)
            new_centroids = self.compute_centroids(centroids, labels)

            '''
            If the K-means algorithm ever results in no change between the inputted centroids and the
            computed centroids, the centroids_changed flag is set to false, signifying that the algorithm
            has converged and the while loop will terminate.
            '''
            if (np.array_equal(centroids, new_centroids)):
                centroids_changed = False
            centroids = new_centroids

        '''
        Return a numpy array of k point coordinates signifying the coverged centroids of the k-clusters in
        the given data.
        '''
        return np.array(centroids)
    
if __name__ == "__main__":
    anchors = get_anchors(root = '/Users/calixkim/VIP27920/bdd100k/')
    print(anchors)
