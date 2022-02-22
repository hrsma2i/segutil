from __future__ import division, annotations

import matplotlib
import numpy as np
from matplotlib import pyplot as plot
from matplotlib.patches import Patch


def voc_colormap(category_ids):
    """Color map used in PASCAL VOC

    Args:
        categories (iterable of ints): Category ids.

    Returns:
        numpy.ndarray: Colors in RGB order. The shape is :math:`(N, 3)`,
        where :math:`N` is the size of :obj:`categories`. The range of the values
        is :math:`[0, 255]`.

    """
    colors = []
    for category_id in category_ids:
        r, g, b = 0, 0, 0
        i = category_id
        for j in range(8):
            if i & (1 << 0):
                r |= 1 << (7 - j)
            if i & (1 << 1):
                g |= 1 << (7 - j)
            if i & (1 << 2):
                b |= 1 << (7 - j)
            i >>= 3
        colors.append((r, g, b))
    return np.array(colors, dtype=np.float32)


def vis_image(img, ax=None):
    """Visualize a color image.

    Args:
        img (~numpy.ndarray): See the table below.
            If this is :obj:`None`, no image is displayed.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)
    if img is not None:
        ax.imshow(img.astype(np.uint8))
    return ax


def vis_segmap(
    img,
    segmap,
    category_names=None,
    category_colors=None,
    ignore_category_color=(255, 255, 255),
    alpha=0.6,
    all_category_names_in_legend=False,
    ax=None,
):
    """Visualize a semantic segmentation.

    Args:
        img (~numpy.ndarray): See the table below. If this is :obj:`None`,
            no image is displayed.
        segmap (~numpy.ndarray): See the table below.
        category_names (iterable of strings): Name of categories ordered according
            to category ids.
        category_colors: (iterable of tuple): An iterable of colors for regular
            cagtegories.
            Each color is RGB format and the range of its values is
            :math:`[0, 255]`.
            If :obj:`colors` is :obj:`None`, the default color map is used.
        ignore_category_color (tuple): Color for ignored category.
            This is RGB format and the range of its values is :math:`[0, 255]`.
            The default value is :obj:`(0, 0, 0)`.
        alpha (float): The value which determines transparency of the figure.
            The range of this value is :math:`[0, 1]`. If this
            value is :obj:`0`, the figure will be completely transparent.
            The default value is :obj:`1`. This option is useful for
            overlaying the category on the source image.
        all_category_names_in_legend (bool): Determines whether to include
            all category names in a legend. If this is :obj:`False`,
            the legend does not contain the names of unused categories.
            An unused category is defined as a category that does not appear in
            :obj:`segmap`.
            The default value is :obj:`False`.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`segmap`, ":math:`(H, W)`", :obj:`int32`, \
        ":math:`[-1, \#category - 1]`"

    Returns:
        matploblib.axes.Axes and list of matplotlib.patches.Patch:
        Returns :obj:`ax` and :obj:`legend_handles`.
        :obj:`ax` is an :category:`matploblib.axes.Axes` with the plot.
        It can be used for further tweaking.
        :obj:`legend_handles` is a list of legends. It can be passed
        :func:`matploblib.pyplot.legend` to show a legend.

    """
    if category_names is not None:
        n_categories = len(category_names)
    elif category_colors is not None:
        n_categories = len(category_colors)
    else:
        n_categories = segmap.max() + 1

    if category_colors is not None and not len(category_colors) == n_categories:
        raise ValueError(
            "The size of category_colors is not same as the number of categories"
        )
    if segmap.max() >= n_categories:
        raise ValueError("The values of segmap exceed the number of categories")

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    if category_names is None:
        category_names = [str(l) for l in range(segmap.max() + 1)]

    if category_colors is None:
        category_colors = voc_colormap(list(range(n_categories)))
    # [0, 255] -> [0, 1]
    category_colors = np.array(category_colors) / 255
    cmap = matplotlib.colors.ListedColormap(category_colors)

    canvas_img = cmap(segmap / (n_categories - 1), alpha=alpha)

    # [0, 255] -> [0, 1]
    ignore_category_color = (np.array(ignore_category_color) / 255,)
    canvas_img[segmap < 0, :3] = ignore_category_color

    ax.imshow(canvas_img)

    legend_handles = []
    if not all_category_names_in_legend:
        legend_categories = [l for l in np.unique(segmap) if l >= 0]
    else:
        legend_categories = range(n_categories)
    for l in legend_categories:
        legend_handles.append(
            Patch(color=cmap(l / (n_categories - 1)), label=category_names[l])
        )

    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1, 1),
        loc=2,
    )

    return ax, legend_handles
