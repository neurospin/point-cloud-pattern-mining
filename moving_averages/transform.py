from soma import aims
import os
from os import path
from typing import Sequence
import numpy
from tqdm import tqdm
import logging
import pathlib
from multiprocessing import Pool, cpu_count
from functools import partial

from .distance import calc_distance

from . import files_utils as fu
from . import distance

log = logging.getLogger(__name__)


def transform_datapoints(data_points, dxyz, rotation_matrix, translation_vector, flip=False):
    """Transform the data_points.

    The datapoint are scaled according to dxyz, then rotated with rotation_matrix and
    translated by translation_vector.

    If flip is True, the x coordinates are inverted (x --> -x)
    """

    tr_data_points = numpy.empty_like(data_points, dtype=float)

    for i, point in enumerate(data_points):
        # multiply by scale factors
        point = point*dxyz

        # make the array a column vector
        # point = point.reshape(1,3)

        # >>> APPLY THE TRANSFORMATION
        # rotate (by matrix multiplication)
        rot_point = numpy.dot(rotation_matrix, point)
        # translate
        transformed_point = rot_point + translation_vector

        if flip:
            transformed_point[0] = -transformed_point[0]

        # round
        # transformed_point = numpy.round(transformed_point)

        # store
        tr_data_points[i] = transformed_point

    return tr_data_points


def talairach_transform(bucket_file: str, transformation_file: str, flip: bool = False):
    log.warn("This function is deprecated, please use load_buket")
    return load_buket(bucket_file, transformation_file, flip)


def load_buket(bucket_file, transformation_file=None, flip=False):
    """Load a buket file and tailerac transform with a given transformation file.

    :param bucket_file: path to the bucket file (.bck)
    :type bucket_file: str
    :param transformation_file: path to the transformation file (.trm)
    :type transformation_file: str
    :param flip: if True the x coordinates are inverted, defaults to False
    :type flip: bool, optional
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

    return data_points, raw_data_points, dxyz, rotation, translation


def talairach_transform_batch(bucket_in_path: str, transformPath: str, flip: bool = False):
    log.warn("This function is deprecated, please use load_bukets")
    return load_bukets(bucket_in_path, transformPath, flip)


def load_bukets(buket_folder: str, transformation_folder: str = None, flip: bool = False):
    """Apply the Tailerach transform to the buckets files in a given folder and return transformed buckets as list of ndarrays.

    :param buket_folder: path that contains the bucket files
    :type buket_folder: str
    :param transformation_folder: path that contains the corresponding transformation files (.trm)
    :type transformation_folder: str
    :param flip: if True the x coordinates are inverted
    :type flip: bool

    return: The transformed bucket ranged in a dictionary with the corresponding bucket filename as keys.
    :rtype: dictionary of numpy.ndarray
    """

    files = os.listdir(buket_folder)

    output_data = dict()
    output_raw = dict()
    output_tra = dict()
    output_rot = dict()
    output_dxyz = dict()

    for bck_file_name in tqdm(files, desc="Talairach Transform" if not flip else "Talairach Transform (flip)"):
        bkt_file_path = path.join(buket_folder, bck_file_name)
        bkt_file_name_base, _ = bck_file_name.split('.')

        subject_name = bkt_file_name_base[1:]  # eliminate L/R prefix

        if transformation_folder is not None:
            transformation_file_path = path.join(
                transformation_folder, subject_name + "_tal.trm")
            assert(path.exists(transformation_folder)), "transformation file not found for {}".format(
                bkt_file_name_base)
        else:
            transformation_file_path = None

        data_points, raw_data_points, dxyz, rotation, translation = load_buket(
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


# TODO : REMOVE THIS FUNCTION
# def talairach_transform_and_save_batch(bucket_in_path: str, transformPath: str,
#                                        binary_out_path: str, bucket_out_path: str,
#                                        flip: bool = False):
#     """Apply the Tailerach transform to the buckets files in a given folder and generates transformed buckets.

#     Two files are generated per bucked, they represent the same data in two different formats: .bck (bucket format) and .ndarray (binary numpy array)
#     3Output folders are created if they do not exist.

#     :param bucket_in_path: path that contains the bucket files
#     :type bucket_in_path: str
#     :param transformPath: path that contains the corresponding transformation files (.trm)
#     :type transformPath: str
#     :param binary_out_path: output path for the binary data
#     :type binary_out_path: str
#     :param bucket_out_path: output path for the transformed bucket files
#     :type bucket_out_path: str
#     :param flip: if True the x coordinates are inverted
#     :type flip: bool
#     """

#     # example of bucket file header:
#     # ascii
#     # -type VOID
#     # -dx 0.796875
#     # -dy 0.796875
#     # -dz 0.796875
#     # -dt 1
#     # -dimt 1
#     # -time 0
#     # -dim 1001
#     # (111 ,106 ,95)
#     # ...
#     #################################################################################

#     bck_header_line_n = 9  # number of lines in the header of the bucket files
#     files = os.listdir(bucket_in_path)

#     # create output folders if needed
#     pathlib.Path(binary_out_path).mkdir(parents=True, exist_ok=True)
#     pathlib.Path(bucket_out_path).mkdir(parents=True, exist_ok=True)

#     for bck_file_name in tqdm(files, desc="Talairach Transform"):

#         # path of the input bucket file
#         bkt_file_path = path.join(bucket_in_path, bck_file_name)
#         bkt_file_name_base, _ = bck_file_name.split('.')

#         # >>> NAME OF THE OUTPUT BINARY FILE
#         # this file contains the transformed bucket and is created with numpy.save
#         array_dump_fname = bkt_file_name_base
#         if flip:
#             array_dump_fname = "flip-" + array_dump_fname
#         array_dump_fname += '.ndarray'
#         array_dump_fname = path.join(binary_out_path, array_dump_fname)

#         # >>> TALAI-BUCKET OUTPUT FILE NAME
#         # a new bucket file (.bck) with the transformed points
#         talai_bck_out_fname = "talai-" + bck_file_name
#         talai_bck_out_fname = path.join(bucket_out_path, talai_bck_out_fname)

#         # >>> FIND THE TRANSFORMATION FILE
#         # from the file corresponding to the subject
#         subject_name = bkt_file_name_base[1:]  # eliminate L/R prefix
#         transformation_file_path = path.join(
#             transformPath, subject_name + "_tal.trm")
#         assert(path.exists(transformPath)), "transformation file not found for {}".format(
#             bkt_file_name_base)

#         data_points = talairach_transform(
#             bucket_file=bkt_file_path,
#             transformation_file=transformation_file_path,
#             flip=flip
#         )

#         fu.save_bucket_file(bucket_out_path, data_points)

#         # >>> WRITE THE TANSFORMED BUCKET TO FILE
#         data_points.tofile(array_dump_fname)
#         pass


def align_buckets_by_ICP(volume_to_align: numpy.ndarray, ref_volume: numpy.ndarray):
    """Align two volumes by icp"""
    # calculate ICP
    _, rot, tra = distance.calc_distance(
        volume_to_align, ref_volume, 'icp',
        max_iter=1e3, epsilon=1e-2)
    # transform the sulcus with the ICP rotation matrix and translation vector
    data_points = transform_datapoints(volume_to_align, (1, 1, 1), rot, tra)

    return data_points, rot, tra


def align_buckets_by_ICP_batch(volumes_dict: Sequence[numpy.ndarray], ref_subject_name: str):
    """ Align all subjects in bucket_dict to a reference subject, using ICP.

    This function is parallelized and uses all available CPUs.

    Args:
        volumes_dict (Sequence[numpy.ndarray]): dictionary of volumes indexed by subject names
        ref_jubject_name (str): name of the reference subject to which the other are aligned

    Returns:
        dict: a dictionary of aligned buckets indexed by subject name
    """
    assert ref_subject_name in volumes_dict

    model_bucket = volumes_dict[ref_subject_name]
    subjects = volumes_dict.keys()
    # TODO make sure the order is the same here
    other_buckets = volumes_dict.values()

    f = partial(align_buckets_by_ICP, ref_volume=model_bucket)

    with Pool(cpu_count()) as p:
        icp_output = list(tqdm(p.imap(f, other_buckets), total=len(other_buckets),
                               desc="Aligning buckets to {}".format(ref_subject_name)))

    distance_matrices = []
    rotation_matrices = []
    translation_vectors = []

    for dist, rot, tra in icp_output:
        distance_matrices.append(dist)
        rotation_matrices.append(rot)
        translation_vectors.append(tra)

    return dict(zip(subjects, distance_matrices)), dict(zip(subjects, rotation_matrices)), dict(zip(subjects, translation_vectors))
