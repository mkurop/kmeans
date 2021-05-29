# kmeans
KMeans algorithm for computing a codebook based on a training set.

# Requirements

* NumPy
* CuPy

# Example use of the class

```
NUMBER_OF_TRAIN_POINTS = 1000000                                                                                                                                                                        
DIMENSION_OF_TRAIN_POINTS = 10                                                                                                                                                                             
REQUESTED_NUMBER_OF_CENTROIDS = 100               

train_set = np.random.randn(DIMENSION_OF_TRAIN_POINTS,NUMBER_OF_TRAIN_POINTS).astype(np.float32)                                                                                                                                                      
initial_codebook = kmeanspp_numpy.kmeanspp(train_set,REQUESTED_NUMBER_OF_CENTROIDS)                                                                                                                                                                           
codebook = KMeans(train_set,initial_codebook)                                                                                                                                                                                                           
cb = codebook.train()         
```

To run the demo, go to the <repo_root>/src directory and invoke

```
python3 kmeans_numpy.py
```

or

```
python3 kmeans_cupy.py
```

for a GPU version of the algorithm.

# Benchmark
Problem size:
* number of codevectors: 100
* dimension of codevectors: 10
* size of the training set: 1mln vectors

CPU wall clock time: 1m35s

NVIDIA TITAN RTX GPU wall clock time: 9s

GPU is more than 10x faster
