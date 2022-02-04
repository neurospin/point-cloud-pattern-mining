import numpy
import pandas
from numpy.core import numeric
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from .transform import align_pcs
import numpy as _np
from tqdm import tqdm as _tqdm


def get_MA_gauss_weigths(coords_df: pandas.DataFrame, center: numeric, FWHM: numeric, normalized=True):
    """ Get the SPAM Gaussian weights for each subject and each axis in isomap_axis_df

    Args:
        coords_df (pandas.DataFrame or numpy.ndarray): the coordinates of each subject,
            generally this is the ouptup of the isomap calculation (all axis)
        center_x (numeric): The center of the SPAM for which the weights are calculated
        FWHM (numeric): Full Width at Half Maximum of the gaussian
        normalized (bool): Normalize the total weights so that the sum is one

    Returns:
        same type as coords_df: SPAM weights
    """
    std = FWHM/2.3548200450309493  # = 2*sqr(2*log(2))
    w = numpy.exp(-(coords_df-center)**2/(2*std**2))
    if normalized:
        w = w/w.sum()
    return w


def _f_round(x): return numpy.round(x).astype(numpy.int64)


def find_closest_subject_on_axis(coordinate, distance_df, axis_n):
    """Find the subject that is closest to the specified coordinate
    in the specified axis of the given distance matrix"""

    return (_np.abs(distance_df.loc[:, axis_n] - coordinate)).idxmin()


def calc_one_MA_volume(buckets_dict, distance_df, axis_n, center, FWHM, min_weight=None, align_to_center=False):
    """Calculate the moving average of the buckets as 3D image, centered in 'center'

    Each point of each bucket is contributes with a weight calculated
    according to the distance from center. The weights are gaussians and
    the width of the gussian is specified with its FWHM.


    Args:
        buckets_dict ([type]): a dictionary of buckets
        distance_df ([type]): dataframe containing the distances (probably the output of the isomap calculation).
            the index values must be the same as the keys of buckets_dict.
        axis_n (int): number of the axis (column) in distance_df to use as distance axis
        center (numeric): the center of the moving average
        FWHM (numeric): Full width at half maximum of the gaussian weighting function
        min_weight (float in [0,1]) : Subjects with weight lower than this are skipped
        if align_to_center is True, for each center, the buckets are realigned to the subject closest subject

    Returns:
        tuple : (MA as volume, shift)
        the shift vector should be summed to the image coordinates to translate the MA in the original space of the buckets.
    """

    # get the weights for all subjects at position "center"
    weigths = get_MA_gauss_weigths(distance_df, center, FWHM).loc[:, axis_n]

    d = buckets_dict

    # TODO transform this filter in a filter function of the FWHM
    # TODO cut subjects that are at more than n standatd devientions
    # eliminate low_weigth subjects
    if (min_weight is not None):
        d = {name: bucket for name, bucket in buckets_dict.items()
             if weigths[name] > min_weight}
        # print(len(buckets_dict), len(d) )

    # alignment
    if align_to_center:
        closest_subj = find_closest_subject_on_axis(
            center, distance_df, axis_n)
        d, _, _ = align_pcs(d, closest_subj, verbose=False)

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

    # per each point in each bucket, add the subject's weight to the corresponding voxel in the volume
    for subject, bucket in d.items():
        bucket = _f_round((bucket - (xmin, ymin, zmin)))
        for x, y, z in bucket:
            assert x >= 0, x
            assert y >= 0, y
            assert z >= 0, z
            vol[x, y, z] += weigths[subject]

    return vol, numpy.round((xmin, ymin, zmin)).astype(int)


def calc_MA_volumes_batch(centers, buckets_dict, distance_df, axis_n, FWHM, min_weight=None):
    """Calulate all moving averages (3D images) at specified centers.

    Arguments:
        min_weight (float in [0,1]) : Subjects with weight lower than this are skipped
        align_to_center : if True, for each center, the buckets are realigned to the subject closest subject
    """

    f = partial(calc_one_MA_volume,
                buckets_dict,
                distance_df,
                axis_n,
                min_weight=min_weight,
                align_to_center=False,  # can not set tu true, because will spawn other subprocesses
                FWHM=FWHM)

    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(f, centers), total=len(centers),
                            desc="Calculating moving averages"))

    volumes, shifts = list(zip(*results))

    return dict(zip(centers, volumes)), dict(zip(centers, shifts))


def calc_MA_volumes_with_alignment(buckets_dict, distance_df, axis_n, centers, FWHM, min_weight=None):
    """calculate the moving averages at the specified centers. 
    Before averaging, per each center, the point-clouds are aligned to the subject witch is closest to the center"""

    volumes = list()
    shifts = list()
    for center in _tqdm(centers):
        volume, shift = calc_one_MA_volume(
            buckets_dict, distance_df, axis_n, center=center, FWHM=FWHM, min_weight=min_weight, align_to_center=True)
        volumes.append(volume)
        shifts.append(shift)

    MA_volumes = dict(zip(centers, volumes))
    MA_shifts = dict(zip(centers, shifts))

    return MA_volumes, MA_shifts


def MA_volumes_to_buckets(volumes_dict, q=0.3, FWHM=1):
    """Transform moving averages volumes into buckets.

    Args:
        volumes_dict (dict): a dictionary containing moving averages volumes
        q (float in [0,1], optional): Quantile of the voxel intensities population.
                The voxels below this quantile will not be represented. Defaults to 0.3.
        FWHM (float, optional): Full widht half maximum of a prealable gaussian blur.
            The blur is the first step of this function. Defaults to 1.

    Returns:
        dict : a dictionary of buckets. The keys are the moving-average centers.
    """
    buckets = dict()
    assert q <= 1 and q >= 0, "q is a quantile, it must be in [0,1]"

    for k, x in volumes_dict.items():
        x = gaussian_filter(x, FWHM)
        xq = numpy.quantile(x[x > 0], q)
        x[x <= xq] = 0
        buckets[k] = numpy.argwhere(x)
    return buckets
