import numpy
from typing import Sequence, Type
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import pandas


def plot_sulci(list_of_sulci: Sequence[numpy.ndarray], labels: Sequence[str] = None, transpose=True, x_shift=0) -> None:

    """ Plot sulci (with plotly Scatter3D).

    :param sulci: a list of arrays representing the sulci to plot
    :type sulci: Sequence[numpy.ndarray]
    :param transpose: transpose each sulcus array if True, defaults to True
    :type transpose: bool, optional
    :param x_shift: Amount of x shift to add to each sulcus, defaults to 0
    :type x_shift: num, optional

    """

    gos = []

    if labels is None:
        labels = [None] * len(list_of_sulci)

    assert len(labels) == len(
        list_of_sulci), "ERROR: len(labels) != len(list_of_sulci)"


    for i, sulcus in enumerate(list_of_sulci):

        if transpose:
            sulcus = sulcus.T

        assert sulcus.shape[0] == 3,\
            "The shape array is not correct, first dimension should be 3, got{}".format(
                sulcus.shape[0])

        x, y, z = sulcus

        try:
            len(x)
        except:
            # x is a number and not iterable
            raise ValueError(
                "Only one point found in the sulcus, make sure you are providing a list of sulci to the plot function.")

        # do not set opacity<1 because of a plotpy issue:
        # https://github.com/plotly/plotly.js/issues/2717
        gos.append(
            go.Scatter3d(x=x+x_shift*i, y=y, z=z, mode='markers',
                         marker=dict(size=1, opacity=1),
                         name=labels[i])
        )

    fig = go.Figure(data=gos)

    # markers in the legend are of fixed (big) size
    fig.update_layout(legend={'itemsizing': 'constant'})

    return fig


def draw_isomap_embedding(embedding: numpy.ndarray or pandas.DataFrame, n_bins=20, density_2D=False):
    """Draw the isomap output in a scatter plot.
    The number of dimensions of the embedding must be from 1 to 3.
            
    Args:
        embedding (numpy.ndarray orpandas.DataFrame): The reduced dimension output of isomap
        n_bins (int, optional): histogram bins. Defaults to 20.
        density_2D (bool, optional): if True display only the first two axis with a 2D histogram. Defaults to False.

    Raises:
        TypeError: [description]
    """

    if isinstance(embedding, pandas.DataFrame):
        data = numpy.array([embedding.iloc[:,i] for i in range(len(embedding.columns))])
    elif isinstance(embedding, numpy.ndarray):
        data = embedding
    else:
        raise TypeError("expected array or DataFrame")


    group_labels = ["dimension {}".format(i+1) for i in range(len(embedding.columns))]
    bin_size = (data.max()-data.min())/n_bins
    
    if density_2D:
        ff.create_2d_density(data[0],data[1], point_size=3).show()
    else:
        ff.create_distplot(data, group_labels, bin_size=bin_size, histnorm=None, show_curve=False).show()


def brochette_layout(plotly_figure, title:str=None, hight=3):
    """Adapt the figure to the 'brochette' plot (many entities with an x shift)."""

    fig = plotly_figure

    fig.layout.xaxis1.update( dict(range=[0, 100]) )

    camera = dict(
        eye=dict(x=0, y=0.5, z=hight)
    )

    fig.update_layout(scene_camera=camera, title=title)
    fig.update_scenes(aspectratio=dict(x=5,y=1,z=1))
    return fig
  