import os
from os import path
from typing import Sequence, Tuple
import numpy as np
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

from . import files_utils as fu
from . import distance

log = logging.getLogger(__name__)


# TODO: USE SCIPY AFFINE_TRANSFORM
def transform_datapoints(data_points: np.ndarray, dxyz: np.ndarray, rotation_matrix: np.ndarray, translation_vector: np.ndarray, flip: bool = False) -> np.ndarray:
    """Transform the data_points.

    The datapoint are scaled according to dxyz, then rotated with rotation_matrix and
    translated by translation_vector.

    If flip is True, the x coordinates are inverted (x --> -x)
    """

    tr_data_points = np.empty_like(data_points, dtype=float)

    for i, point in enumerate(data_points):
        # multiply by scale factors
        point = point*dxyz

        if (rotation_matrix is not None) and not np.array_equal(rotation_matrix, np.eye(3)):
            # print("rotate", point)
            point = np.dot(rotation_matrix, point)

        # translate
        if (translation_vector is not None) and not np.array_equal(translation_vector, np.zeros(3)):
            # print("translate", point, translation_vector)
            point = point + translation_vector

        if flip:
            # print(point)
            point[0] = -point[0]

        # store
        tr_data_points[i] = point

    return tr_data_points


def talairach_transform(bucket_file: str, transformation_file: str, flip: bool = False):
    log.warn("This function is deprecated, please use load_bucket")
    return load_bucket(bucket_file, transformation_file, flip)


def load_bucket(bucket_file: str, transformation_file: str = None, flip: bool = False) -> Tuple:
    """Load a bucket file and tailerac transform with a given transformation file.

    :param bucket_file: path to the bucket file (.bck)
    :param transformation_file: path to the transformation file (.trm)
    :param flip: if True the x coordinates are inverted, defaults to False
    :return: data_points_tr, data_points_raw, rotation, translation
    :rtype: 
    """
    # >>> PARSE THE BUCKET FILE
    dxyz, data_points, dim = fu.parse_bucket_file(bucket_file)
    raw_data_points = data_points.copy()
    rotation = None
    translation = None

    # >>> THE TAILERACH TRANSFORM HAPPENS HERE

    if transformation_file is not None:
        # get the rotation matrix and tranlation vector
        rotation, translation = fu.parse_transformation_file(
            transformation_file)
        data_points = transform_datapoints(
            data_points=data_points,
            dxyz=dxyz,
            rotation_matrix=rotation,
            translation_vector=translation,
            flip=flip
        )
    else:
        # apply the scaling even if the transformation is not given
        rotation = np.eye(3)  # identity
        translation = np.zeros(3)  # [0,0,0]
        data_points = transform_datapoints(
            data_points=data_points,
            dxyz=dxyz,
            rotation_matrix=rotation,
            translation_vector=translation,
            flip=flip
        )

    return data_points, raw_data_points, dxyz, rotation, translation


def talairach_transform_batch(bucket_in_path: str, transformPath: str, flip: bool = False):
    log.warn("This function is deprecated, please use load_buckets")
    return load_buckets(bucket_in_path, transformPath, flip)


def load_buckets(bucket_folder: str, transformation_folder: str = None, flip: bool = False):
    """Apply the Tailerach transform to the buckets files in a given folder and return transformed buckets as list of ndarrays.

    :param bucket_folder: path that contains the bucket files
    :type bucket_folder: str
    :param transformation_folder: path that contains the corresponding transformation files (.trm)
    :type transformation_folder: str
    :param flip: if True the x coordinates are inverted
    :type flip: bool

    return: The transformed bucket ranged in a dictionary with the corresponding bucket filename as keys.
    :rtype: dictionary of numpy.ndarray
    """

    files = os.listdir(bucket_folder)

    output_data = dict()
    output_raw = dict()
    output_tra = dict()
    output_rot = dict()
    output_dxyz = dict()

    for bck_file_name in tqdm(files, desc="Talairach Transform" if not flip else "Talairach Transform (flip)"):
        bkt_file_path = path.join(bucket_folder, bck_file_name)
        bkt_file_name_base, _ = bck_file_name.split('.')

        subject_name = bkt_file_name_base[1:]  # eliminate L/R prefix

        if transformation_folder is not None:
            transformation_file_path = path.join(
                transformation_folder, subject_name + "_tal.trm")
            assert(path.exists(transformation_folder)), "transformation file not found for {}".format(
                bkt_file_name_base)
        else:
            transformation_file_path = None

        data_points, raw_data_points, dxyz, rotation, translation = load_bucket(
            bucket_file=bkt_file_path,
            transformation_file=transformation_file_path,
            flip=flip
        )

        output_data[bkt_file_name_base] = data_points
        output_raw[bkt_file_name_base] = raw_data_points
        output_tra[bkt_file_name_base] = rotation
        output_rot[bkt_file_name_base] = translation
        output_dxyz[bkt_file_name_base] = dxyz

    return output_data, output_raw, output_dxyz, output_tra, output_rot


def align_pc_pair(pc_to_align: np.ndarray, reference_pc: np.ndarray,
                  max_iter=1e3, epsilon=1e-2):
    """Align two point-clouds by ICP"""
    # calculate ICP
    _, rot, tra = distance.calc_distance(
        pc_to_align, reference_pc, 'icp',
        max_iter=max_iter, epsilon=epsilon)
    # transform the bucket with the ICP rotation matrix and translation vector
    data_points = transform_datapoints(pc_to_align, (1, 1, 1), rot, tra)

    return data_points, rot, tra


def align_pcs(pcs: Sequence[np.ndarray], reference_pc_name: str, cores=None, verbose=True):
    """ Align the given point-clouds to a reference point-cloud, using ICP.

    This function is parallelized, it uses all CPUs minus 3 by default, if cores is not specified.

    Args:
        pcs (Sequence[np.ndarray]): dictionary of point-clouds indexed by subject names
        reference_pc_name (str): name of the reference point-cloud to which the other are aligned.
            It must be in the keys of pcs

    Returns:
        a list of 3 dicionnaries indexed by subject name, containing per each entry:
            the aligned point-clouds
            the rotation matrix
            the translation vector
    """
    assert reference_pc_name in pcs

    model_bucket = pcs[reference_pc_name]
    subjects = pcs.keys()
    # TODO make sure the order is the same here
    other_pcs = pcs.values()

    f = partial(align_pc_pair, reference_pc=model_bucket)

    if cores is None:
        cores = cpu_count()-3

    if verbose:
        log.info(f"using {cores} cores out of {cpu_count()}")

        with Pool(cores) as p:
            icp_output = list(tqdm(p.imap(f, other_pcs), total=len(other_pcs),
                                   desc="Aligning point-clouds to {}".format(reference_pc_name)))
    else:
        with Pool(cores) as p:
            icp_output = p.map(f, other_pcs)

    point_clouds = []
    rotation_matrices = []
    translation_vectors = []

    for dist, rot, tra in icp_output:
        point_clouds.append(dist)
        rotation_matrices.append(rot)
        translation_vectors.append(tra)

    return dict(zip(subjects, point_clouds)), dict(zip(subjects, rotation_matrices)), dict(zip(subjects, translation_vectors))


def sort_buckets(buckets: dict, isomap_df, axis: int):
    """Sort the buckets according to their distance along the specified isomap axis."""
    sorted_subjects = isomap_df.loc[:, axis].sort_values().keys()
    return {subj: buckets[subj] for subj in sorted_subjects}
