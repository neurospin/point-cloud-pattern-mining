import argparse
import os
from glob import glob
from tqdm import tqdm
import dico_toolbox as dtb
import numpy as np
from multiprocessing import Pool, cpu_count

import logging
log = logging.getLogger(__name__)

try:
    from soma import aims
except:
    ImportError(
        "pyAims could not be imported. Cannot use this tool outside a brainvisa environment.")


def get_fname(path): return os.path.basename(path).split('.')[0]


def volume_to_points_colud(path):
    """convert a volume into a point-colud."""

    fname = get_fname(path)

    try:
        vol = aims.read(path)
        point_cloud = dtb.convert.volume_to_bucket_numpy(vol)
        error_mgs = None
    except Exception as e:
        point_cloud = None
        error_mgs = str(e)

    return {"name": fname, "point-cloud": point_cloud, "error_mgs": error_mgs}


def main(*args, **kwargs):
    parser = argparse.ArgumentParser(
        description="Convert specified AIMS volume files into point clouds and store them in one compressed numpy file.")
    parser.add_argument(
        "input_path", help="The path to the volume to convert (wildcards are admitted m e.g. *.nii)", nargs='*', type=str)
    parser.add_argument("-o", "--output_path",
                        help="The path of the output file containing the bucket", type=str)
    args = parser.parse_args()

    out_path = args.output_path
    if out_path is None:
        out_path = "point_clouds.npz"
    base_out_dir = os.path.dirname(out_path)

    # check output path validity
    if not os.path.isdir(base_out_dir):
        log.critical(f'"{base_out_dir}" is not a valid directory')
        return 1

    fun = volume_to_points_colud

    with Pool(cpu_count() - 3) as pool:
        out = list(tqdm(
            pool.imap(fun, args.input_path), total=len(args.input_path)))

    pcs = {d['name']: d['point-cloud'] for d in out if d['error_mgs'] is None}
    errors = [d['error_mgs'] for d in out if d['error_mgs'] is not None]

    print("Creating output file...", end='')

    np.savez_compressed(out_path, **pcs)
    print("Done.")

    if len(errors) > 0:
        error_str = [f"There were {len(errors)} ERRORS:", *errors]
        log.error("\n".join(error_str))
