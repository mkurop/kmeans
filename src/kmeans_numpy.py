#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kmeans_numpy.py
=============================================================================
Implements the kmeans algorithm for finding an optimal codebook. Uses kmeans++\
as initialization algorithm.
"""

import random
import site
import os

site.addsitedir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../kmeanspp/src"))
import kmeanspp_numpy
import numpy as np

class KMeans:

    """ Class for training of the codebook based on a training set and an initial codebook. Uses the\
    KMeans algorithm.

    :param train_set: training set with training vectors as columns, the computations precision is inherided from the\
    numeric type of the training vectors
    :type train_set: np.ndarray
    :param initial_codebook: initial codebook, prepared using e.g. KMeans++ algorith, centroids as columns
    :type initial_codebook: np.ndarray
    """
    def __init__(self, train_set : np.ndarray, initial_codebook : np.ndarray):
        """Constructor method"""

        self.train_set = train_set

        # The number of training points is assumed to be larger than dimension \
        # of the training points

        # force training points to be columns
        self._train_set = np.transpose(train_set) if train_set.shape[0] > train_set.shape[1] else train_set

        # check dimension of the initial_codebook
        if not self._train_set.shape[0] in set([initial_codebook.shape[0], initial_codebook.shape[1]]):
            raise ValueError("Dimension of codevectors in the inial codebook is unproper...")

        # transpose the inital codebook if needed, force the numeric type of initial codebook to be\
        # the same like the type of the training set
        self._initial_codebook = initial_codebook.astype(train_set.dtype) if initial_codebook.shape[0] == self._train_set.shape[0] else \
                                               initial_codebook.T.astype(train_set.dtype)
        self._codebook = self._initial_codebook

        self._M = self._train_set.shape[1]

        self._d = self._train_set.shape[0]

        self._K = self._initial_codebook.shape[1]

        self._distortion = np.Inf


    def _permute(self, sortin, permutation):

        assert len(sortin) == len(permutation)
        return [sortin[i] for i in permutation]

    def _my_sort(self, inputlist):

        inputlist = zip(inputlist, range(len(inputlist)))
        aux = sorted(inputlist, key=lambda x: x[0])
        sorted2in = [aux[i][1] for i in range(len(aux))]
        list2 = zip(sorted2in, range(len(sorted2in)))
        aux1 = sorted(list2, key=lambda x: x[0])
        in2sorted = [aux1[i][1] for i in range(len(aux1))]
        sort = [aux[i][0] for i in range(len(aux))]
        return sort, in2sorted, sorted2in

    def _get_diffs(self):
        # below statement results in KxMxd tensor, K - number of codevectors, M- number of training points\
        # d - dimension of the train points
        self._diffs = self._codebook[np.newaxis,...].transpose((2,0,1)) - \
                                               self.train_set[np.newaxis,...].transpose((0,2,1))

    def _get_distances_squared(self):

        self._distances_squared = np.sum(self._diffs * self._diffs, axis = 2)

    def _assign_train_vectors_to_voronoi_regions(self):

        self._idxs = np.argmin(self._distances_squared, axis = 0)

    def _get_distortion(self):

        self._distortion_prev = self._distortion

        self._distortion = np.sqrt(np.sum(np.amin(self._distances_squared,axis = 0))/(self._M*self._d))

    def _update_codebook(self):

        self._number_of_train_points_per_voronoi_region = np.zeros((self._codebook.shape[1],))

        self._empty_voronoi_regions = []

        for k in range(self._K):
            
            self._number_of_train_points_per_voronoi_region[k] = np.sum(self._idxs == k)

            if not self._number_of_train_points_per_voronoi_region[k] == 0:
                                                
                self._codebook[:,k] = 1/self._number_of_train_points_per_voronoi_region[k] * np.sum(self._train_set[:,self._idxs == k], axis = 1)

            else:

                self._empty_voronoi_regions.append(k)

                                                
    def _handle_empty_voronoi_regions(self):
        
        aux1, in2sorted, sorted2in = self._my_sort(self._number_of_train_points_per_voronoi_region)

        i = 1

        AUX_MULTIPLICATOR = 2

        while np.sum(aux1[-i:]) < AUX_MULTIPLICATOR * len(self._empty_voronoi_regions):

            i += 1

        candidates_for_replacing_centroids_of_empty_cells = self._train_set[:,self._idxs == sorted2in[-1]]

        for j in range(1,i):

            candidates_for_replacing_centroids_of_empty_cells = np.concatenate((candidates_for_replacing_centroids_of_empty_cells, self._train_set[:, self._idxs == sorted2in[-j]]), axis = 1)

        # get candidate train_points to replace the centroids of emtpy Voronoi regions

        samples = random.sample(list(range(candidates_for_replacing_centroids_of_empty_cells.shape[1])), len(self._empty_voronoi_regions))

        # replace the centroids

        for i, sample in enumerate(samples):

            self._codebook[:,self._empty_voronoi_region[i]] = self._train_set[:,sample]
            

    def train(self, convergence_criterion : float = 1e-4) -> np.ndarray:
        """The main method of the class. The method trains actually the codebook.

        :param convergence_criterion: convergence criterion (D_prev - D)/D < convergence_criterion, defaults to 1e-4
        :type convergence_criterion: float
        :return: the method returs the trained codebook with centroids as columns
        :rtype: np.ndarray
        """

        iteration_num = 0

        while True:

            self._get_diffs()

            self._get_distances_squared()

            self._get_distortion()

            if (self._distortion_prev - self._distortion)/self._distortion < convergence_criterion:

                return self._codebook

            print(f"Distortion per training vector per dimension at iteration {iteration_num} is: {self._distortion}")

            iteration_num += 1

            self._assign_train_vectors_to_voronoi_regions()

            
            self._update_codebook()

            if self._empty_voronoi_regions:

                self._handle_empty_voronoi_regions()

            
if __name__ == "__main__":

    NUMBER_OF_TRAIN_POINTS = 1000000

    DIMENSION_OF_TRAIN_POINTS = 10

    REQUESTED_NUMBER_OF_CENTROIDS = 100

    train_set = np.random.randn(DIMENSION_OF_TRAIN_POINTS,NUMBER_OF_TRAIN_POINTS).astype(np.float32)
    
    initial_codebook = kmeanspp_numpy.kmeanspp(train_set,REQUESTED_NUMBER_OF_CENTROIDS)     

    codebook = KMeans(train_set,initial_codebook)

    cb = codebook.train()





            






        

        





