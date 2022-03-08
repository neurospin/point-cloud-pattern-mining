import pandas
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as _np
from . import functions
from typing import Sequence

import logging
log = logging.getLogger(__name__)


class ICP_result:
    def __init__(self, distances, rotations, translations):
        self.dist = distances
        self.rotations = rotations
        self.translations = translations
        self.names = distances.index.values.tolist()
        self.dist_min = _np.minimum(distances, distances.T)
        self.dist_max = _np.maximum(distances, distances.T)

    def __repr__(self):
        return f"ICP result containing {len(self.names)} entries"


# a map of distance types and the corresponding distance calculating function
_distance_types_map = {
    "simple": functions.simple,
    "icp": functions.icp
}

distance_types = list(_distance_types_map.keys())


def _get_distance_f(x):
    if not callable(x):
        # get the appropriate distance function from the distance argument
        try:
            distance_f = _distance_types_map.get(x)
        except:
            raise ValueError(
                "The distance type must be one of {}".format(distance_types))
    return distance_f


def calc_distance(a1: _np.ndarray, a2: _np.ndarray, distance_f: str = 'icp', **kwargs):
    """Calculate the distance between arrays a1 and a2.
    The distance type is specified in the distance_f argument, it must be a string or a function.
    kwargs are passed to the distance function.
py
    The distance functions implemented in this module can be found in sulci_isomap.distance.functions

    Args:
        a1 (_np.ndarray): first array of points (NxD where D is the point dimensions number)
        a2 (_np.ndarray): second array of points (MxD)
        distance_f (str, optional): type of distance. Defaults to 'icp'

    Raises:
        ValueError: if the distance type does not correspond to an implemented distance function

    Returns:
        The return type depends on the function specified in distance_f.
    """

    distance_f = _get_distance_f(distance_f)

    return distance_f(a1.T, a2.T, **kwargs)


def calc_all_distances(point_clouds: Sequence[_np.ndarray],
                       distance_f: str,
                       indexes: Sequence[int] = 'all',
                       n_cpu_max=None, **kwargs):
    """Calculate the distance between point-clouds.

    This fucntion is parallelized and will use multiple CPUs.

    For each index i in indexes, calculate the distance between
    sulci[i] and all other sulci.
    """

    # get the distance function
    distance_f = _get_distance_f(distance_f)

    assert all([s.shape[1] == 3 for s in point_clouds]),\
        "input point-clouds must be arrays of point (expected shape Nx3)"

    # Transpose each pc (get shape 3xN)
    sulci = [s.T for s in point_clouds]

    # the number of CPUs to use
    if n_cpu_max is None:
        n_cpu_max = cpu_count()
    else:
        n_cpu_max = min(cpu_count(), n_cpu_max)

    if indexes == 'all':
        # calc distances for all pcs
        indexes = range(len(sulci))

    dist_results = list()

    for i in tqdm(indexes, desc="Calculating distances"):
        sulcus = sulci[i]
        f = partial(distance_f, sulcus, **kwargs)
        with Pool() as p:
            this_res = p.map(f, sulci)
            dist_results.append(this_res)

    # create a custom datatype numpy array to store the
    # list conveniently
    dt = _np.dtype([
        ('distance', _np.float),
        ('rotation', _np.float, (3, 3)),
        ('translation', _np.float, (3))
    ])
    dist_results = _np.array(dist_results, dtype=dt)

    return dist_results


def calc_all_icp(point_clouds: dict, n_cpu_max: int = None):
    """"Run icp between all pairs of sulci

    :param sulci:point_cloud dictionnary (M, Nx3)
    :type sulci: dict of str:np.array
    :param n_cpu_max: max number of CPUs used, defaults to None (all available)
    :type n_cpu_max: int, optional
    :return: distance matrix, rotations, translations
    :rtype: list of _np.ndarray (NxNx1, NxNx3x3, NxNx3)
    """

    names = list(point_clouds.keys())

    d = calc_all_distances(
        point_clouds.values(), distance_f='icp', n_cpu_max=n_cpu_max)

    # ICP_result(a['distance'], a['rotation'], a['translation'])

    dist = pandas.DataFrame(d['distance'], index=names, columns=names)
    rot = dict(zip(names, d['rotation']))
    tra = dict(zip(names, d['translation']))

    return ICP_result(dist, rot, tra)


def find_MAD_outliers(distance_df: pandas.DataFrame, sd_factor: int = 3):
    """Find outliers based on absolute deviation from the median (MAD).

    :param distance_df: Distance dataframe (symmetric),
        the columns names are considered subjects names
    :type distance_df: pandas.DataFrame
    :param sd_factor: tolerance (in units of absolute median deviantion), defaults to 3
    :type sd_factor: int, optional
    :return: listo of outlier and valid subject names
    :rtype: (list, list)
    """
    # the columns names are considered to be the subjects names
    subjects = distance_df.columns

    # per sulcus: sum of the max distances from oll other sulci
    sum_distances = distance_df.sum(axis=1)

    # median
    med_sum_dist = _np.median(sum_distances)
    # difference from the median
    deltas = _np.abs(sum_distances - med_sum_dist)
    # median of the difference with the median
    med_deltas = _np.median(deltas)
    # distance threshold to discard outliers
    distance_th = med_sum_dist + sd_factor*med_deltas

    outliers_mask = sum_distances <= distance_th

    outlier_names = subjects[_np.logical_not(outliers_mask)]
    valid_names = subjects[outliers_mask]

    return outlier_names.tolist(), valid_names.tolist()

# TODO: outliers par pourcentage de population
