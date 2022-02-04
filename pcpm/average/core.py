
from re import A
from typing import Sequence
from matplotlib.axes import Axes
import numpy
import pandas
from scipy.fftpack import shift
from scipy.stats import norm


class Average_result:
    def __init__(self, vol, offset, n):
        self.n = n
        self.vol = vol
        self.offset = offset

    def __repr__(self):
        return f"Average of {self.n} point-clouds"


def _f_round(x):
    """Round and convert to int64"""
    return numpy.round(x).astype(numpy.int64)


def eucl_dist(p1, p2) -> float:
    """Euclidean distance"""
    return numpy.sqrt(numpy.sum((p1 - p2)**2))


def eucl_norm(p) -> float:
    """Euclidean norm"""
    return numpy.sqrt(numpy.sum(p**2))


def _get_weights(embedding: pandas.DataFrame, center: str or Sequence[float or int], FWHM: float):
    """Calculate Gaussian weights for the given points based on their distance from the center.

    Args:
        embedding (pandas.DataFrame): a data frame containing the coordinates of each point
        center (str or Sequence[float or int]): a valid label of points or a set of coordinate
    """

    std = FWHM/2.3548200450309493  # = 2*sqr(2*log(2))

    if isinstance(center, str):
        center = embedding[center]
    else:
        center = numpy.array(center)

    embedding = pandas.DataFrame(embedding)

    assert (embedding.values[0].shape == center.shape
            ), f"dimension mismatch between center {center.shape}"\
        f"and the other points {embedding.values[0].shape}."

    w = embedding.copy()
    w = w-center
    w = w.apply(eucl_norm, axis=1)
    w = w.apply(lambda x: norm.pdf(x/std))
    w = numpy.sqrt(2*numpy.pi) * w

    return w


def average_pcs_w(point_clouds: dict, weights: Sequence[float]) -> Average_result:
    """Return a weighted average of the given point-clouds."""

    d = point_clouds

    # Create a volume that includes all buckets

    x = numpy.concatenate([d[s][:, 0] for s in d.keys()])
    y = numpy.concatenate([d[s][:, 1] for s in d.keys()])
    z = numpy.concatenate([d[s][:, 2] for s in d.keys()])

    xmin = x.min()
    ymin = y.min()
    zmin = z.min()

    xmax = x.max()
    ymax = y.max()
    zmax = z.max()

    shape = _f_round(numpy.array((xmax-xmin+1, ymax-ymin+1, zmax-zmin+1)))
    vol = numpy.zeros(shape)

    # add the weight to the corresponding voxel in the volume
    for name, weight in weights.items():
        pc = point_clouds[name]
        pc = _f_round((pc - (xmin, ymin, zmin)))
        for x, y, z in pc:
            assert x >= 0, x
            assert y >= 0, y
            assert z >= 0, z
            vol[x, y, z] += weights[name]

    return Average_result(vol, numpy.round((xmin, ymin, zmin)).astype(int), n=len(point_clouds))


# def average_pcs(point_clouds: pandas.DataFrame, embedding: pandas.DataFrame, FWHM: float,
#                 center: str or Sequence[float] = None) -> Average_result:
#     """Return a weighted average of the given point-clouds."""

#     names = point_clouds.keys()
#     embedding = embedding.loc[names]

#     weights = _get_weights(embedding, center, FWHM)

#     return average_pcs_w(point_clouds, weights)


def average_pcs(point_clouds: dict, embedding: pandas.DataFrame, FWHM: float,
                cluster_n: int, labels: Sequence[int], centers: Sequence) -> Average_result:
    """Return a weighted average of the given point-clouds."""

    names = embedding.loc[labels == cluster_n].index.values
    embedding = embedding.loc[names]

    weights = _get_weights(embedding, centers[cluster_n], FWHM)

    return average_pcs_w(point_clouds, weights)


# def average_pc_clusters(point_clouds: pandas.DataFrame, embedding: pandas.DataFrame,
#                         labels: Sequence[str], centers):
#     """Return the averages of the clusters of point clouds."""

#     for
