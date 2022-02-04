

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
    """Return lists of names (from the embedding) separated according to the labels.

    Labels are usually the result of a clustering algorithm."""
    names_lists = []
    names = embedding.index.values.tolist()

    namelabels = list(zip(names, labels))

    for label in set(labels):

        names_lists.append(
            [name for name, cur_label in namelabels if cur_label == label])

    return names_lists


def split_pcs_in_clusters(pcs, embedding, labels):
    names_lists = labels_to_names(embedding, labels)
    return [{name: pcs[name] for name in names} for names in names_lists]
