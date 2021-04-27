from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils.graph import graph_shortest_path
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from pandas import DataFrame
import pandas

import logging
log = logging.getLogger(__name__)


class Isomap_dist(Isomap):
    """A modified version of scipi.manifold.isomap that allows distance dataframe as input.

    This class inherits from and beheaves mostly like sklearn.manifold.isomap.Isomap. See the
    corresponding doc for reference.

    Differencies:
        - the n_job parameters is set to -1 by default (use all CPUs)   
        - the default metrics is set to 'precomputed' so data are assumed to be a distance matrix
        - the fit_transform() fuction accepts a distance dataframe as input
    """

    def __init__(self, *, n_jobs=-1, metric='precomputed', **kwargs):
        super().__init__(**kwargs)

        # Number of CPUs to use in calculations (-1 = all available)
        # sklearn defaults to None (= 1 CPU) here we default to -1
        self.n_jobs = n_jobs
        # "precomputed" will be used in the different steps to specify that the
        # input consists in a distance matrix and not in a feature table
        # (for which the distance needs to be calculated)
        self.metric = metric

    def fit_transform_from_df(self, distance_df: pandas.DataFrame):
        """Fit the model from data in distance_df and transform distance_df.

        This function wrapps IsoMap.fit_transform() and allows pandas.DataFrame as input parameter.
        Otherwise it beheaves identically.

        Args:
            distance_df (pandas.DataFrame): Training data, shape = (n_samples, n_samples) The inter-subjects distace data.
            The index of the DataFrame should consist in the name of the subjects.

        Returns:
            pandas.DataFrame : shape=(n_samples, n_components) the isomap output. The index is the same as the inpunt.
        """

        if not isinstance(distance_df, pandas.DataFrame):
            raise ValueError(
                "Expected a pandas.Dataframe, got {}".format(type(distance_df)))

        out = super().fit_transform(distance_df.values)

        return DataFrame(out, index=distance_df.index, columns=range(1, out.shape[1]+1))


    def fit_transform_old(self, X):
        """Calculate the isomap embedding of distance_df
        This function was used in the previous versions of the sulci isomap
        pipeline. The main reason was that the IsoMap object in scikit-learn
        did not accept distance matrices as input.

        Since version 0.22, scikit-learn Isomap supports distance matrices as input
        hence the built-in fit_transform() should be used.

        Args:
            X (numpy.DandarraytaFrame): Training data, a square distance matrix shape = (n_samples, n_samples)
                The inter-subjects distace data.

        Returns:
            numpy.ndarray shape=(n_subjects, n_components): the isomap output
        """

        log.warn("fit_transform_old() function is deprecated and should not be used.")


        # k-nearest neighmours (kNN)
        self.nbrs_ = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm='auto',
            metric=self.metric,
            n_jobs=self.n_jobs)
        self.nbrs_.fit(X)

        # Weighted neigbours graph
        kng = kneighbors_graph(
            self.nbrs_,
            self.n_neighbors,
            mode='distance',
            metric=self.metric,
            n_jobs=self.n_jobs)

        # Perform a shortest-path graph search
        # get the sortest distances matrix within the graph
        self.dist_matrix_ = graph_shortest_path(
            kng,
            method=self.path_method,
            directed=False
        )

        #  Dimensionality reduction
        self.kernel_pca_ = KernelPCA(
            n_components=self.n_components,  # defined at object initialization
            kernel=self.metric,
            eigen_solver=self.eigen_solver,
            tol=self.tol,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs  # -1 = use all available processors
        )

        # WHY THIS?
        G = self.dist_matrix_ ** 2
        G *= -0.5

        return self.kernel_pca_.fit_transform(G)
