"""
Visualization functions of various types: graph data, 3D models, etc.
All of these should somehow return what they draw in a form that can be stored on disk.
"""

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import numpy as np

def plot(datas, name, xlabels=None, xaxis="x", yaxis="y", logscale_x=False, logscale_y=True, legends=None, colors=None, alphas=None, plot_to_screen=True):
    """
    Plot torch data.
    
    Arguments:
        datas -- list with N-length data to plot (1D numpy arrays)

    Keyword arguments:
        [xlabels] -- list with N-length indices on the x-axis (1D numpy arrays)
        [xaxis] -- name of the x axis (defaults to 'x')
        [yaxis] -- name of the x axis (defaults to 'y')
        [logscale_y] -- whether the y axis is drawn in log-scale. Defaults to True.
        [legends] -- the legend for this figure (list of strings). Missing or None entries are ignored.
        [plot_to_screen] -- whether to display this plot on the screen. Defaults to True. Blocks.

    Returns:
        figure -- the matplotlib figure handle
    """

    figure = plt.figure()
    for k in range(len(datas)):
        ys = datas[k]
        xs = xlabels[k]
        if xs is None:
            xs = np.range(ys.size)
        kwargs = {}
        if legends is not None and len(legends) > k:
            kwargs['label'] = legends[k]
        if colors is not None and len(colors) > k:
            kwargs['color'] = colors[k]
        if alphas is not None and len(alphas) > k:
            kwargs['alpha'] = alphas[k]
        plt.plot(xs, ys, **kwargs)
    figure.legend()
    plt.title(name)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    if logscale_x:
        plt.xscale('log')
    if logscale_y:
        plt.yscale('log')
    if plot_to_screen:
        plt.show()
    return figure
