import sys
import os
parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt
from utils import plot_progress_kMeans


class kMenas():
    def __init__(self):
        pass
    
    def _find_closest_centroids(self, X, centroids):
        """
        Computes the centroid memberships for every example
        
        Args:
            X (ndarray): (m, n) Input values      
            centroids (ndarray): (K, n) centroids
        
        Returns:
            idx (array_like): (m,) closest centroids
        """
        K = centroids.shape[0]
        idx = np.zeros(X.shape[0], dtype=int)

        for i in range(X.shape[0]):
            distance = []
            for j in range(K):
                # Compute L2-norm
                norm_ij = np.linalg.norm(X[i]-centroids[j])
                distance.append(norm_ij)
            # Find index of min distance
            idx[i] = np.argmin(distance)
        return idx
    
    def _compute_centroids(self, X, idx, K):
        """
        Returns the new centroids by computing the means of the 
        data points assigned to each centroid.
        
        Args:
            X (ndarray):   (m, n) Data points
            idx (ndarray): (m,) Array containing index of closest centroid for each 
                        example in X. Concretely, idx[i] contains the index of 
                        the centroid closest to example i
            K (int):       number of centroids
        
        Returns:
            centroids (ndarray): (K, n) New centroids computed
        """
        centroids = np.zeros((K, X.shape[1]))

        for k in range(K):
            points = X[idx==k]
            # Recalculate coordinates of centroid
            centroids[k] = np.mean(points, axis=0) 
        return centroids
    
    def _kMeans_init_centroids(self, X, K):
        """
        This function initializes K centroids that are to be 
        used in K-Means on the dataset X
        
        Args:
            X (ndarray): Data points 
            K (int):     number of centroids/clusters
        
        Returns:
            centroids (ndarray): Initialized centroids
        """
        # Randomly reorder the indices of examples
        randidx = np.random.permutation(X.shape[0])
        # Take the first K examples as centroids
        centroids = X[randidx[:K]]
        return centroids
    
    def _run_kMeans(self, X, initial_centroids, max_iters=10, plot_progress=False):
        """
        Runs the K-Means algorithm on data matrix X, where each row of X
        is a single example
        """
        # Initialize values
        K = initial_centroids.shape[0]
        centroids = initial_centroids
        previous_centroids = centroids    
        idx = np.zeros(X.shape[0])
        plt.figure(figsize=(8, 6))

        # Run K-Means
        for i in range(max_iters):
            
            # Output progress
            print("K-Means iteration %d/%d" % (i, max_iters-1))
            # For each example in X, assign it to the closest centroid
            idx = self._find_closest_centroids(X, centroids)
            
            # Optionally plot progress
            if plot_progress:
                plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
                previous_centroids = centroids
                
            # Given the memberships, compute new centroids
            centroids = self._compute_centroids(X, idx, K)
        plt.show() 
        return centroids, idx
    
    def main(self, X, K, max_iters, plot_progress=True):
        # Set initial centroids by picking random examples from the dataset
        initial_centroids = self._kMeans_init_centroids(X, K)
        # Run K-Means
        centroids, idx = self._run_kMeans(X, initial_centroids, max_iters, plot_progress)
        return centroids, idx
    
