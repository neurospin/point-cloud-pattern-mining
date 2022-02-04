from scipy.stats import norm
import pandas
import numpy
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings("ignore", module=".*pandas*")


def _is_iterable(x):
    isIterable = True
    try:
        iter(x)
    except:
        isIterable = False
    return isIterable


def isomap_embedding(isomap: pandas.Series, axes=None, figsize=None, **kwargs):

    embedding = isomap

    if axes is not None:
        embedding = isomap.loc[:, axes]

    embedding_dim = len(embedding.shape)

    # "The centers must have the same dimensionality of the embedding"

    if embedding_dim == 1:
        # mono dimensional embedding

        hits, bins, _ = plt.hist(embedding, **kwargs)
        plt.xlabel(f"axis {axes}")
        plt.ylabel(f"hits")

        figsize = kwargs.get("figsize")
        if figsize is not None:
            plt.gcf().set_size_inches(figsize)

    else:
        if figsize is None:
            figsize = (7, 7)

        kwargs["figsize"] = figsize
        pandas.plotting.scatter_matrix(embedding, **kwargs)


def weights_for_moving_averages(centers, FWHM, max):
    if centers is not None:
        # plot the centers and gaussians representing the weights
        assert all([not _is_iterable(c) for c in centers]
                   ), "All centers must have dimension = 1"

        std = FWHM/2.53
        for center in centers:
            x = numpy.arange(center-3*std, center+3*std)
            plt.plot(x, max*(numpy.sqrt(2*3.14))
                     * norm.pdf((x - center)/std), '--r')

        plt.plot([], [], '--', label="weigths")
        plt.plot(centers, numpy.zeros(len(centers)), 'or', label="centers")


def clustering(data: pandas.DataFrame, labels, cmap='Accent', **kwargs):
    df = pandas.DataFrame(data)
    cmap = matplotlib.cm.get_cmap(cmap, labels.max())
    df['label'] = labels
    df['color'] = list(map(cmap, labels))

    embedding_dim = embedding_dim = len(data.shape)

    if embedding_dim == 1:
        for label in set(labels):
            plt.hist(df.loc[df.label == label].iloc[:, 0], **kwargs)
    elif embedding_dim == 2:
        df.plot.scatter(1, 2, c=df.loc[:, "color"])
    else:
        raise NotImplementedError(
            "Not available for embedding of dimension > 2")
