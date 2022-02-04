from ..isomap import Isomap_dist as _Isomap_dist
import numpy as _numpy
import pandas as _pandas


def get_isomap(data: _pandas.DataFrame, neighbors: int = None, components: int = 3) -> _pandas.DataFrame:
    """Apply the ISOMAP algorithm to the data.

    Args:
        data (_pandas.DataFrame): input data
        neighbors (int, optional): Number of neighbors to consider for each point. If none, its value is log(len(data))
        components (int, optional): Number of coordinates for the manifold.. Defaults to 3.

    Returns:
        _pandas.DataFrame: the
    """

    if neighbors is None:
        neighbors = _numpy.round(_numpy.log(len(data))).astype(int)

    # initialize the isomap class
    isomap_obj = _Isomap_dist(
        n_neighbors=neighbors,
        n_components=components  # number of output PCA components
    )

    # fit the model and transform the data
    isomap_df = isomap_obj.fit_transform_from_df(data)

    isomap_df.sort_index().head()

    return isomap_df
