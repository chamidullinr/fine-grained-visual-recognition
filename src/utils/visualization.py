import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch


def create_fig(*, ntotal=None, ncols=1, nrows=1, colsize=8, rowsize=6, **kwargs):
    if ntotal is not None:
        nrows = int(np.ceil(ntotal / ncols))
    fig, ax_mat = plt.subplots(
        nrows, ncols, figsize=(colsize*ncols, rowsize*nrows), **kwargs)
    axs = np.array(ax_mat).flatten()
    if ntotal is not None and len(axs) > ntotal:
        for ax in axs[ntotal:]:
            ax.axis('off')
    if len(axs) == 1:
        axs = axs[0]
    return fig, axs


def heatmap(df, *, cmap='Blues', fmt='.3f', cbar=False, ax=None,
            colsize=1.2, rowsize=0.7):
    import seaborn as sns
    
    if ax is None:
        fig, ax = create_fig(
            ncols=1, nrows=1,
            colsize=colsize * df.shape[1],
            rowsize=rowsize * df.shape[0])

    # ignore non-numerical values
    df = df.copy()
    # to prevent errors when index has bool values (e.g. after crosstab)
    if df.index.is_boolean():
        df.index = df.index.astype(str)
    if df.columns.is_boolean():
        df.columns = df.columns.astype(str)
    df = df[df.dtypes.index[df.dtypes != 'object']]

    ax = sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap, ax=ax, cbar=cbar)
    ax.tick_params(axis='y', labelrotation=0)
    return ax


def plot_training_progress(df, x='epoch', y='train_loss', title='Training Progress',
                           xlim=[0, 30], ylim=None, annot=False, annot_nth=5, ax=None):
    import seaborn as sns

    assert isinstance(x, str)
    assert x in df
    if isinstance(y, str):
        y = [y]
    for _y in y:
        assert _y in df

    if ax is None:
        fig, ax = create_fig(ncols=1, nrows=1)

    # create lineplot
    # df.plot(x=x, y=y, kind='line', marker='.',
    #         xlim=xlim, ylim=ylim, title=title, ax=ax)
    for _y in y:
        sns.lineplot(data=df, x=x, y=_y, marker='o', label=_y, ax=ax)
        ax.set(xlim=xlim, ylim=ylim, title=title)
        ax.legend()

    ax.grid()

    if annot:
        # add anotations
        idx = np.arange(len(df))
        cond = (idx % annot_nth == 0) | (idx == idx.max())
        for _, row in df[cond].iterrows():
            for metric in y:
                if ylim is None or row[metric] < ylim[1]:
                    ax.text(row[x], row[metric], f'{row[metric]:.2f}',
                            va='bottom', ha='center', fontsize='large')

    return ax


def plot_bbox(bbox, ax=None, c='red', linestyle='-', ms=10, **kwargs):
    if ax is None:
        fig, ax = create_fig(colsize=3, rowsize=3)

    xmin, ymin, xmax, ymax = bbox
    ax.plot(
        [xmin, xmax, xmax, xmin, xmin],
        [ymin, ymin, ymax, ymax, ymin],
        c=c, linestyle=linestyle, ms=ms, **kwargs)  # marker='.',

    return ax


def imshow(img, *, bbox=None, ax=None,
           title=None, axis_off=True, **kwargs):
    if isinstance(img, torch.Tensor):
        assert len(img.shape) == 3
        img = img.detach().cpu().numpy().transpose(1, 2, 0)

    if ax is None:
        fig, ax = create_fig(colsize=3, rowsize=3)

    ax.imshow(img, **kwargs)
    if bbox is not None:
        plot_bbox(bbox, ax=ax)
    if title is not None:
        ax.set(title=title)
    if axis_off:
        ax.axis('off')

    return ax
