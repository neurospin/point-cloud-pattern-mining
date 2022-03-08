from multiprocessing.sharedctypes import Value
import numpy
import pandas


def subset_embedding_by_label(embedding, labels, label=None):

    assert len(embedding) == len(
        labels), "Labels and embedding must have the same length"
    embedding = embedding.loc[labels == label]

    return embedding


def find_central_pcs_name(embedding: pandas.DataFrame, labels=None, cluster_label=None):
    """Get the name of the central subject.

    The center subject is the one that is the closest to all others.

    Args:
        embedding (pandas.DataFrame): the embedding of the point-clouds
        labels ([Sequence], optional): [description]. an array of labels for the points of the embedding
        cluster_label ([type], optional): a specific label to filter the embedding. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        [str]: The name of the central point_cloud
    """

    if labels is not None:
        assert len(embedding) == len(labels)
        if cluster_label is None:
            raise ValueError("Please specify the cluster to use.")

        embedding = subset_embedding_by_label(embedding, labels, cluster_label)

    # distances from the mean coordinate (sum over columns)
    sum_of_dist = (numpy.abs(embedding - embedding.mean()) ** 2).sum(axis=1)
    # The center subject is the one that is the less far from
    # all the others
    return sum_of_dist.sort_values().index[0]


def find_closest_pcs_name(coordinates, embedding, labels=None, cluster_label=None):
    """Given an embedding, returns the name of the closest point cloud to the given coordinates.

    Args:
        embedding (pandas.DataFrame): the embedding of the point-clouds
        labels ([Sequence], optional): [description]. an array of labels for the points of the embedding
        cluster_label ([type], optional): a specific label to filter the embedding. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        [str]: The name of the closest point_cloud
    """

    if labels is not None:
        assert len(embedding) == len(labels)
        if cluster_label is None:
            raise ValueError("Please specify the cluster to use.")

        embedding = subset_embedding_by_label(embedding, labels, cluster_label)

    # per each point in the given embedding
    # caluclate the distances too point specified by the given coordinates
    # return the name of the minimimum distance element
    return embedding.apply(lambda x: numpy.sum((x-coordinates)**2), axis=1).idxmin()
