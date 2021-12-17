# Point-cloud-pattern-mining

This library can be used to compare entities described as sets of 3D point clouds (like cortical sulci or fibers bundles).

The analysis starts with the initial calculation of a geometrical distance among the different N point clouds (aka subjects), resulting in a N by N distance matrix. Then multidimensional scaling via isomap is applied to infer the relationships that link entities sharing common features. The isomap algorithm estimates the intrinsic geometry of the data manifold based on a rough estimate of each data point’s neighbors. Then, the geodesic distances among the subjects are recalculated on the inferred manifold. The top n eigenvectors of the geodesic distance matrix represent the coordinates in the new n-dimensional Euclidean space.

The point clouds are then studied in this new n-dimensional space. Nevertheless, direct observation of all the point-clouds is not simple, especially if the number of subjects is large. In order to make a synthetic description of the analysis result, a set of weighted averages are calculated along a chosen axis (see figure 1) or on given regions of the newly defined n-dimensional space, possibly specified by a clustering algorithm.

## install
clone this repo somewhere where you will keep it, then in the root folder install with `pip3`:

```
$ cd point-colud-pattern-mining
$ pip3 install -e .

```

the `-e` options causes the install to link the module to the cloned folder, instead of installing it in the default module location.
