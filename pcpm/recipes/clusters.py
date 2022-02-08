

from typing import Sequence
import pandas


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


def split_pcs_in_clusters(pcs, embedding, labels) -> dict:
    """Split the point cloud ditcionnary according to the given labels

    Args:
        pcs ([type]): [description]
        embedding ([type]): [description]
        labels ([type]): [description]

    Returns:
        dict: a label:dict dictionary of point cloud dictionaries. Labels are converted into strings.
    """

    assert len(labels) == len(
        embedding), "Labels and embedding musth have the same length"

    names_lists = labels_to_names(embedding, labels)
    return {str(label): {name: pcs[name] for name in names} for label, names in names_lists.items()}
