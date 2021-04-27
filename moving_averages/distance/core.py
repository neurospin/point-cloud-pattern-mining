import pandas
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy
from . import functions
from typing import Sequence

import logging
log = logging.getLogger(__name__)


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


def calc_distance(a1: numpy.ndarray, a2: numpy.ndarray, distance_f: str = 'icp', **kwargs):
    """Calculate the distance between arrays a1 and a2.
    The distance type is specified in the distance_f argument, it must be a string or a function.
    kwargs are passed to the distance function.
py
    The distance functions implemented in this module can be found in sulci_isomap.distance.functions

    Args:
        a1 (numpy.ndarray): first array of points (NxD where D is the point dimensions number)
        a2 (numpy.ndarray): second array of points (MxD)
        distance_f (str, optional): type of distance. Defaults to 'icp'

    Raises:
        ValueError: if the distance type does not correspond to an implemented distance function

    Returns:
        The return type depends on the function specified in distance_f.
    """

    distance_f = _get_distance_f(distance_f)

    return distance_f(a1.T, a2.T, **kwargs)


def calc_all_distances(sulci: Sequence[numpy.ndarray],
                       distance_f: str,
                       indexes: Sequence[int] = 'all',
                       n_cpu_max=None, **kwargs):
    """Calculate the distance between sulci.

    This fucntion is parallelized and will use multiple CPUs.

    For each index i in indexes, calculate the distance between
    sulci[i] and all other sulci.
    """

    # get the distance function
    distance_f = _get_distance_f(distance_f)

    assert all([s.shape[1] == 3 for s in sulci]),\
        "input sulci must be arrays of point (expected shape Nx3)"

    # Transpose each sulcus (get shape 3xN)
    sulci = [s.T for s in sulci]

    # the number of CPUs to use
    if n_cpu_max is None:
        n_cpu_max = cpu_count()
    else:
        n_cpu_max = min(cpu_count, n_cpu_max)

    if indexes == 'all':
        # calc distances for all sulci
        indexes = range(len(sulci))

    dist_results = list()

    for i in tqdm(indexes, desc="Calculating distances"):
        sulcus = sulci[i]
        f = partial(distance_f, sulcus, **kwargs)
        with Pool() as p:
            this_res = p.map(f, sulci)
            dist_results.append(this_res)

    return dist_results


def calc_all_icp(sulci: Sequence[numpy.ndarray], n_cpu_max: int = None):
    """"Run icp between all pairs of sulci

    :param sulci: list of sulci (M, Nx3)
    :type sulci: Sequence[numpy.ndarray]
    :param n_cpu_max: max number of CPUs used, defaults to None (all available)
    :type n_cpu_max: int, optional
    :return: distance matrix, rotations, translations
    :rtype: list of numpy.ndarray (NxNx1, NxNx3x3, NxNx3)
    """
    d = calc_all_distances(sulci, distance_f='icp', n_cpu_max=n_cpu_max)
    dt = numpy.dtype([
        ('distance', numpy.float),
        ('rotation', numpy.float, (3, 3)),
        ('translation', numpy.float, (3))
    ])
    a = numpy.array(d, dtype=dt)

    return a['distance'], a['rotation'], a['translation']


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
    med_sum_dist = numpy.median(sum_distances)
    # difference from the median
    deltas = numpy.abs(sum_distances - med_sum_dist)
    # median of the difference with the median
    med_deltas = numpy.median(deltas)
    # distance threshold to discard outliers
    distance_th = med_sum_dist + sd_factor*med_deltas

    outliers_mask = sum_distances <= distance_th

    outlier_names = subjects[numpy.logical_not(outliers_mask)]
    valid_names = subjects[outliers_mask]

    return outlier_names.tolist(), valid_names.tolist()

# TODO: outliers par pourcentage de population

def get_center_subject(distances_df:pandas.DataFrame):
    """Get the name of the central subject.
    
    The center subject is the one that is closer to all the others.

    :param distances_df: a symmetric distance DataFrame,
        the index contains the subjects names
    :type distances_df: pandas.DataFrame
    :return: the name of the center subject
    :rtype: str
    """
    # distances from all other elements (sum over columns)
    sum_of_dist = distances_df.sum(axis=1)
    # The center subject is the one that is the less far from
    # all the others
    return sum_of_dist.sort_values().index[0]