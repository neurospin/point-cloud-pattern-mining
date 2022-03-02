
from re import A
from typing import Sequence
from matplotlib.axes import Axes
import numpy
import pandas
from scipy.fftpack import shift
from scipy.stats import norm
from ..transform import align_pcs
from ..embedding import find_central_pcs_name


class Average_result:
    """Store the result of averaging a set of point-clouds"""

    def __init__(self, vol, offset, n, rotation=None, translation=None, coord_in_embedding=None):
        self.n = n
        self.vol = vol
        self.offset = offset
        self.translation = translation
        self.rotation = rotation
        self.coord_in_embedding = coord_in_embedding

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
        assert center in embedding.index, "center not in embedding"
        center = embedding.loc[center]
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


def average_pcs_w(point_clouds: dict, weights: Sequence[float], normalize: bool) -> Average_result:
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
            vol[x, y, z] += weight

    if normalize:
        vol = (vol-vol.min())/(vol.max()-vol.min())

    return Average_result(vol, numpy.round((xmin, ymin, zmin)).astype(int), n=len(point_clouds))


def average_pcs(point_clouds: dict, embedding: pandas.DataFrame, align_to: str, FWHM: float, normalize: bool = True) -> Average_result:
    """Return a weighted average of the given point-clouds.

    Weights in [0,1] are calculated with 1-dimensional Gaussian function of the distance (in the embedding) of each pc to the given reference.


    Args:
        point_clouds (dict): [description]
        embedding (pandas.DataFrame): [description]
        align_to (str): [description]
        FWHM (float): [description]
        normalize (bool, optional): map the output voxel values to [0,1]. Defaults to True. Defaults to True.

    Returns:
        Average_result: [description]
    """

    names = list(point_clouds.keys())

    assert align_to in names, f"The name '{align_to}' is not in the point cloud keys"

    embedding = embedding.loc[names]

    weights = _get_weights(embedding, align_to, FWHM)

    av = average_pcs_w(point_clouds, weights, normalize=normalize)
    av.coord_in_embedding = embedding.loc[align_to].values
    return av


def average_each_cluster(clusters, embedding, FWHM, centers='auto', align=True, normalize=True):
    """Generate the averages for each of the given clusters

    Args:
        clusters (Sequence): a label:dict(pcs) dictionary of clusters of point clouds
        embedding (DataFrame): [description]
        centers (dict, optional): a label:name dict specifying the reference pc for each cluster, to be considered as the center of the gaussian weights.
            Defaults to 'auto': the central subject is used (min dist from others).      
        align (bool, optional): if True, the cluster is aligned to the point-cloud which is closest to the center.      
        normalize (bool, optional): map the output voxel values to [0,1]. Defaults to True.
    Returns:
        Average_result: a class containing the result of the average calculation
    """

    labels = list(clusters.keys())

    if centers == 'auto':
        # use the central point_cloud
        references = {label: find_central_pcs_name(
            embedding.loc[clusters[label].keys()], labels=None, cluster_label=None) for label in set(labels)}
    else:
        assert clusters.keys() == labels.keys(), "Labels and clusters must have the same keys"

    if (references is not None) and align:
        # align each cluster
        align_result = {label: align_pcs(
            clusters[label], references[label]) for label in labels}
        pcs = {label: align_result[label][0] for label in labels}
        rotations = {label: align_result[label][1] for label in labels}
        translations = {label: align_result[label][2] for label in labels}
    else:
        pcs = clusters
        rotations = None
        translations = None

    av_results = {}

    for label in labels:
        av_results[label] = average_pcs(pcs[label], embedding=embedding,
                                        align_to=references[label], FWHM=FWHM, normalize=normalize)

        av_results[label].rotation = rotations[label] if rotations is not None else None
        av_results[label].translation = translations[label] if translations is not None else None

    return av_results
