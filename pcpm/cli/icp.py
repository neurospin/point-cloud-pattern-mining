import pcpm as ma
import argparse

import numpy as np
from multiprocessing import cpu_count
import os
import pandas as pd
from random import sample
import logging
log = logging.getLogger(__name__)

# custom packages


def distances_by_icp(npz_path, n=None, jobs=None, epsilon=0.1, max_iter=10):
    """Calculate distance by ICP.
    Return the distance_matrix and the final rotations matrix and translations vectors used during distance calculation."""

    data = dict(np.load(npz_path))
    names = [str(k) for k in list(data.keys())]

    if n is not None:
        names = sample(names, n)

    if jobs is None:
        jobs = cpu_count() - 3

    # ALIGN THE POINTS CLOUDS AND CALCULATE ICP DISTANCE
    pcs = {name: data[name] for name in names}
    icp = ma.calc_all_icp(pcs, n_cpu_max=jobs,
                          epsilon=epsilon, max_iter=max_iter)
    dist_df = pd.DataFrame(icp.dist, index=names, columns=names)
    return dist_df, icp.rotations, icp.translations


def main(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Calculate the distance matrix via ICP. Outputs a NPZ file containing distance matrix,\
        and the final rotations matrix and translations vectors used during distance calculation.\
        \
        Example: pcpm_icp point_cloud.npz -o icp/ -j 46 -n 800")
    parser.add_argument(
        "input_path", help="Path to a compressed numpy NPZ file where the point clouds are stored", type=str)
    parser.add_argument("-o", "--output_folder",
                        help="The folder where the output file will be saved, defauts to current folder.", type=str, default='.')
    parser.add_argument(
        "-n", "--n_samples", help="Size of subsample. Is specified, the distance is calculated only for n randomly chosen subjects", type=int)
    parser.add_argument(
        "-j", "--jobs", help="Number of parallel jobs for the ICP calculation", type=int)
    parser.add_argument(
        "-e", "--epsilon", help="min distance improvement on one iteration to continue. Defaluts to 0.1", type=float, default=0.1)
    parser.add_argument(
        "-i", "--max_iter", help="Max number of iteration. Defaults to 10.", type=int, default=10)
    parser.add_argument(
        "-f", "--distance_function", help="The distance function One of {}. Defaults to {}".format(ma.distance.core.distance_types, ma.distance.core.DEFAULT_ICP_FUNCTION), type=str)
    parser.add_argument(
        "-c", "--config_file", help="path to a YAML configuration file (only if libpointmatcher is available). \
            If a configuration file is set, the parameters `max_iter` and `epsilon` are ignored.", type=str)
    args = parser.parse_args()

    # create output folder if it does not exist
    if not os.path.isdir(args.output_folder):
        log.info(
            f'"directory {args.output_folder}" does not exist and will be created.')
        os.makedirs(args.output_folder, exist_ok=True)

    print()
    print("ICP function used: {}".format(ma.distance.DEFAULT_ICP_FUNCTION))

    if ma.distance.libpointmatcher.HAS_LIBPOINTMATCHER:
        if args.config_file:
            ma.distance.libpointmatcher.set_default_ICP_object_from_yaml(
                args.config_file)

        print()
        print("PIPELINE INFORMATION:")
        print()
        print(ma.distance.libpointmatcher.string_of_icp_object(
            ma.distance.libpointmatcher.get_default_icp_object()
        ))

    dist_df, rots, tras = distances_by_icp(
        args.input_path, n=args.n_samples, jobs=args.jobs, epsilon=args.epsilon, max_iter=args.max_iter)

    fname = os.path.basename(args.input_path).split('.')[0]

    dist_df.to_csv(os.path.join(args.output_folder,
                                f"{fname}_distances.csv"))
    np.savez_compressed(os.path.join(args.output_folder,
                        f"{fname}_transformations.npz"), rots=rots, tras=tras)
