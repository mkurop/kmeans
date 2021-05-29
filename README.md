# kmeans
KMeans algoritym for computing a codebook based on a traning set. Running on GPU or CPU.

# Requirements

* NumPy

# Example use of the class

```
NUMBER_OF_TRAIN_POINTS = 1000000                                                                                                                                                                        
DIMENSION_OF_TRAIN_POINTS = 10                                                                                                                                                                             
REQUESTED_NUMBER_OF_CENTROIDS = 100                                                                                                                                                                                                      │~                             
train_set = np.random.randn(DIMENSION_OF_TRAIN_POINTS,NUMBER_OF_TRAIN_POINTS).astype(np.float32)                                                                                                                                         │~                             
initial_codebook = kmeanspp_numpy.kmeanspp(train_set,REQUESTED_NUMBER_OF_CENTROIDS)                                                                                                                                                      │~                             
codebook = KMeans(train_set,initial_codebook)                                                                                                                                                                                            │~                             
cb = codebook.train()         
```

To run the demo, got to the <repo_root>/src directory and invoke

```
python3 kmeans_numpy.py
```
