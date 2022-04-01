

from collections import Counter
from typing import Sequence
import pandas
from ..transform import transform_datapoints


def split_embedding_clusters(embedding: pandas.DataFrame, labels: Sequence[str]):
    df = embedding.copy()
    df['labels'] = labels
    return [x for _, x in df.groupby('labels')]


# def split_pcs_clusters(points: dict, labels: Sequence[str]):
#     """Split the given point-cloud dict into cluster according to the labels."""

#     out = []

#     for label in set(labels):
#         [name for name, cur_label in namelabels if cur_label == label])


#     return [{k: v for k, v in points_label.items() if k == label} for label in set(labels)]


def labels_to_names(embedding: pandas.DataFrame, labels: Sequence[str]) -> Sequence[str]:
    """Return a label:names dictionary (from the embedding) separated according to the labels.

    Labels are usually the result of a clustering algorithm."""
    names_lists = {}
    names = embedding.index.values.tolist()

    namelabels = list(zip(names, labels))

    for label in set(labels):

        names_lists[label] = [name for name,
                              cur_label in namelabels if cur_label == label]

    return names_lists


def rename_labels_by_count(labels):
    """Rename the given label arrays, sorting labels by item counts

    Args:
        labels: array of labels (e.g. the output of clustering)

    Returns:
        array of labels
    """
    counter = Counter(labels)
    d = {item[0]: i for i, item in enumerate(
        sorted(counter.items(), key=lambda x: x[1], reverse=True))}
    new_labels = list(map(lambda x: d[x], labels))

    return new_labels


def split_pcs_in_clusters(pcs, embedding, labels, reorder_labels=True) -> dict:
    """Split the point cloud ditcionnary according to the given labels.

    If reorder_labels is true, the labels are renamed (0 to len(clusters))
    so that cluster 0 is the biggest, 1 the second biggest and so on.

    Args:
        pcs ([type]): [description]
        embedding ([type]): [description]
        labels ([type]): [description]

    Returns:
        dict: a label:dict dictionary of point cloud dictionaries. Labels are converted into strings.
    """

    assert len(labels) == len(
        embedding), "Labels and embedding musth have the same length"

    if (reorder_labels):
        labels = rename_labels_by_count(labels)

    names_lists = labels_to_names(embedding, labels)
    return {str(label): {name: pcs[name] for name in names} for label, names in names_lists.items()}


# TODO: write an alin_cluster function

def apply_affine3D_to_cluster(cluster, rotations, translations):
    transformed_clusters = dict()
    for name, pc in cluster.items():
        transformed_clusters[name] = transform_datapoints(
            pc, dxyz=(1, 1, 1), rotation_matrix=rotations[name], translation_vector=translations[name])

    return transformed_clusters


def get_names_of_n_largest_clusters(clusters, n):
    # get the clusters size
    d = {k: len(cluster) for k, cluster in clusters.items()}
    # sort the clusters
    sorted_names = [e[0] for e in sorted(
        d.items(), key=lambda item:item[1], reverse=True)]
    return sorted_names[:n]


def sort_clusters_by_counts(clusters):
    """Sort and re-label the cluster by the count of their elements (from largest to smallest)"""
    out = dict()
    ordered_names = get_names_of_n_largest_clusters(clusters, len(clusters))
    for i, name in enumerate(ordered_names):
        out[str(i)] = clusters[name]
    return out


def get_n_largest_clusters(clusters, n):
    """Filter clusters to keep only the n largest

    Args:
        clusters (dict): clusters of point-clouds
    """

    cluster_names = get_names_of_n_largest_clusters(clusters, n)
    out = {k: clusters[k] for k in cluster_names}
    others = {x: clusters[k][x]
              for k in clusters if k not in cluster_names for x in clusters[k]}
    return out, others
