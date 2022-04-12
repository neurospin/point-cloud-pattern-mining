import numpy
import pandas
from os import path as _path
from re import A, match as _re_match
from ..distance.core import ICP_result
from typing import Sequence


def load_pcs_from_npz(filename: str) -> dict:
    """Load point-clouds from a compressed numpy array (npz).

    This kind of file (npz) is the typical output of the `pcpm_volume_to_point_cloud` command line tool

    Args:
        filename (str): the path to the .npz file

    Returns:
        dict: a {name:point-cloud} dictionary
    """
    return dict(numpy.load(filename))


def load_icp_result(csv_path: str, npz_path: str = None) -> dict:
    """Load distances and transformation matrices that result from ICP.

    These results are the output of the `pcpm_icp` command line tool.
    They are contained in the specified folder 

    Args:
        csv_path (str): path to the csv file containing the distances
        npz_path (str): path to the npz file containing the transformations.
            If None, it is assumed to be located in the same folder of the csv file.

    Returns:
        dict: [description]
    """

    if npz_path is None:
        # set the path of the transformation file
        dirname, basename = _path.split(csv_path)
        prefix = _re_match(r"(.*)_distances.csv", basename).groups()[0]
        fname = f"{prefix}_transformations.npz"
        if prefix is None:
            raise FileNotFoundError(
                f"The file {fname} should be in the same folder of {basename}")
        npz_path = _path.join(dirname, fname)

    # load distance matrix
    dist = pandas.read_csv(csv_path, index_col=0)
    # load transformations
    rots, trans = dict(numpy.load(npz_path, allow_pickle=True)).values()

    return ICP_result(dist, rots, trans)


def save_point_clouds_as_npz(pcs: dict, path: str) -> None:
    """Save the given point clouds as compressed numpy array (.npz)

    Args:
        pcs (dict): a name:ndarray dictionnary containing the point-clouds
        path (str): the path of the output file
    """
    numpy.savez_compressed(path, **pcs)


def subset_of_pcs(pcs: dict, names: Sequence[str]) -> dict:
    """Get a subset of the point-clouds corresponding to the given names.

    Args:
        pcs (dict): point-clouds
        names (list[str]): list of names for the new subset

    Returns:
        dict: the point-clouds corresponding to the specified names
    """
    return {name: pcs[name] for name in names}


def subset_of_distances(distances: pandas.DataFrame, names: Sequence[str]) -> pandas.DataFrame:
    """Return a distance matric corresponding to the specified names

    Args:
        distances (pandas.DataFrame): distance DataFrame
        names (list[str]): list of name to select in the distance DataFrame

    Returns:
        pandas.DataFrame: Distance DataFrame for the given names
    """
    return distances.loc[names, names]
