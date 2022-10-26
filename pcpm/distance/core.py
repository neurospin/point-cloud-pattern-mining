import dis
import pandas
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as _np
from . import functions, libpointmatcher
from typing import Sequence

import logging
log = logging.getLogger(__name__)


try:
    from pypointmatcher import pointmatcher
    HAS_LIBPOINTMATCHER = True
except ImportError as e:
    HAS_LIBPOINTMATCHER = False


class Distance_result:
    def __init__(self, distances, rotations, translations):
        self.dist = distances
        self.rotations = rotations
        self.translations = translations
        self.names = distances.index.values.tolist()
        self.dist_min = _np.minimum(distances, distances.T)
        self.dist_max = _np.maximum(distances, distances.T)

    def __repr__(self):
        return f"Distance result containing {len(self.names)} entries"


# a map of distance types and the corresponding distance calculating function
_distance_types_map = {
    "simple": functions.simple,
    "icp_python": functions.icp_python,
    "coarse_PCA": functions.coarse_PCA,
}


_DEFAULT_DIST_FUNCTION_NAME = "icp_python"


if HAS_LIBPOINTMATCHER:
    _distance_types_map["icp_libpointmatcher"] = libpointmatcher.icp_libpointmatcher
    _DEFAULT_DIST_FUNCTION_NAME = "icp_libpointmatcher"


distance_types = list(_distance_types_map.keys())


def _get_distance_f(f):
    if not callable(f):
        # get the appropriate distance function from the distance argument
        if f not in distance_types:
            # log.warning("The distance name must be one of {}. Got {}".format(
            #     distance_types, f))
            # log.warning("deafulting to {}".format(_DEFAULT_DIST_FUNCTION_NAME))
            f = _DEFAULT_DIST_FUNCTION_NAME
        # get the specified distance function, default to the PYTHON implementation
        distance_f = _distance_types_map.get(f)

    return distance_f


def get_DEFAULT_DIST_FUNCTION_NAME():
    """get the name of the default distance function."""
    global _DEFAULT_DIST_FUNCTION_NAME
    return _DEFAULT_DIST_FUNCTION_NAME


def set_DEFAULT_DIST_FUNCTION_NAME(f):
    """Set the default distance function.
    Args:
        function can be a either a callable or a function name.
    """

    global _DEFAULT_DIST_FUNCTION_NAME
    _DEFAULT_DIST_FUNCTION_NAME = f


def calc_distance(a1: _np.ndarray, a2: _np.ndarray,
                  distance_f: str = _DEFAULT_DIST_FUNCTION_NAME, **kwargs):
    """Calculate the distance between arrays a1 and a2.
    The distance type is specified in the distance_f argument, it must be a string or a function.
    kwargs are passed to the distance function.
py
    The distance functions implemented in this module can be found in sulci_isomap.distance.functions

    Args:
        a1 (_np.ndarray): first array of points (NxD where D is the point dimensions number)
        a2 (_np.ndarray): second array of points (MxD)
        distance_f (str, optional): type of distance. Defaults to 'icp'
        kwargs are passed to the distance function

    Raises:
        ValueError: if the distance type does not correspond to an implemented distance function

    Returns:
        The return type depends on the function specified in distance_f.
    """

    distance_f = _get_distance_f(distance_f)

    return distance_f(a1, a2, **kwargs)


def calc_all_distances(point_clouds: Sequence[_np.ndarray],
                       distance_f: str = _DEFAULT_DIST_FUNCTION_NAME,
                       indexes: Sequence[int] = 'all',
                       n_cpu_max=None, **kwargs):
    """Calculate the distance between point-clouds.

    This fucntion is parallelized and will use multiple CPUs.

    For each index i in indexes, calculate the distance between
    point-cloud[i] and all other point-clouds.

    kwargs are passed to the distance function.

    """

    # get the distance function
    distance_f = _get_distance_f(distance_f)

    assert all([s.shape[1] == 3 for s in point_clouds]),\
        "input point-clouds must be arrays of point (expected shape Nx3)"

    pcs = [s for s in point_clouds]

    # the number of CPUs to use
    if n_cpu_max is None:
        n_cpu_max = cpu_count() - 3
    else:
        n_cpu_max = min(cpu_count(), n_cpu_max)

    if indexes == 'all':
        # calc distances for all pcs
        indexes = range(len(pcs))

    dist_results = list()

    for i in tqdm(indexes, desc="Calculating distances (with {})".format(distance_f.__name__)):
        pc = pcs[i]
        f = partial(distance_f, pc, **kwargs)
        with Pool() as p:
            this_res = p.map(f, pcs)
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


def calc_all_icp(point_clouds: dict, n_cpu_max: int = None,
                 distance_f=_DEFAULT_DIST_FUNCTION_NAME, max_iter=10, epsilon=0.1, **kwargs):
    """"Run ICP over all the pairs of point-clouds.

    Args:
        point_clouds: dictionnary (M, Nx3) of str:ndarray
        sulci: dict of str:np.array
        n_cpu_max: max number of CPUs used, defaults to None (all available)
        n_cpu_max: int, optional
        epsilon (float, optional): min distance improvement on one iteration. Defaults to 0.1.
        max_iter (int, optional): stop at this iteration. Defaults to 10.
        kwargs are directly passed to the distance function


    Returns:
        list of np.ndarray (NxNx1, NxNx3x3, NxNx3): distance matrix, rotations, translations

    """

    names = list(point_clouds.keys())

    if distance_f == "icp_libpointmatcher":
        libpointmatcher.set_icp_object_parameters(
            epsilon=epsilon, max_iter=max_iter)

    d = calc_all_distances(
        point_clouds.values(), distance_f=distance_f, n_cpu_max=n_cpu_max, max_iter=max_iter, epsilon=epsilon, **kwargs)

    # ICP_result(a['distance'], a['rotation'], a['translation'])

    dist = pandas.DataFrame(d['distance'], index=names, columns=names)
    rot = dict(zip(names, d['rotation']))
    tra = dict(zip(names, d['translation']))

    return Distance_result(dist, rot, tra)


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
