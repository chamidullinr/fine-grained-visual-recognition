import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from src.utils import io, visualization as viz


def load_specs_files(path):
    """Load specification JSON files and store them as pandas DataFrame."""
    filenames = io.get_filenames_in_dir(path, filetype='json')
    specs = [io.read_json(item) for item in filenames]
    specs_df = pd.DataFrame(specs)

    # sort by date
    assert 'history_file' in specs_df
    specs_df['date'] = specs_df['history_file'].str[-20:-4]
    specs_df = specs_df.sort_values('date').reset_index(drop=True)

    return specs_df


def load_progress_files(specs_df, path, history_file_col='history_file'):
    """Load training progress CSV files and concatenate them into one pandas DataFrame."""
    assert 'history_file' in specs_df
    out = []
    for _, row in specs_df.iterrows():
        filename = os.path.join(path, row[history_file_col])
        if os.path.isfile(filename):
            _df = pd.read_csv(filename)
            for col in specs_df.columns:
                if not isinstance(row[col], (list, tuple)):
                    _df[col] = row[col]
            out.append(_df)
    out = pd.concat(out, ignore_index=True)
    return out


def filter_items(df, *, outlen=None, copy=False, **kwargs):
    """
    Filter records in pandas DataFrame.

    Filtering is done by **kwargs parameters.
    """
    for k, v in kwargs.items():
        assert k in df, f'Key "{k}" is missing in the dataframe'
        df = df[df[k] == v]
    if outlen is not None:
        assert len(df) == outlen, f'Got group lenth: {len(df)}; expected: {outlen}'
    if copy:
        df = df.copy()
    return df


def aggtime(arr, aggfunc='mean'):
    """
    Aggregate time series.
    """
    from datetime import datetime, timedelta
    import time

    def str2seconds(x):
        try:
            dt = datetime.strptime(x, '%H:%M:%S')
        except ValueError:
            dt = datetime.strptime(x, '%M:%S')
        tdelta = timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second)
        return tdelta.total_seconds()

    def seconds2str(x):
        minute, second = divmod(x, 60)
        if second > 0: # ceil minutes
            minute += 1
        hour, minute = divmod(minute, 60)
        # day, hour = divmod(hour, 24)
        # if day > 0:
        #      out = f'{day:.0f}-{hour:.0f}:{minute:02.0f}:{second:02.0f}'
        # else:
        #     out = f'{hour:.0f}:{minute:02.0f}:{second:02.0f}'
        if hour > 0:
            out = f'{hour:.0f}h {minute:02.0f}m'
        else:
            out = f'{minute:02.0f}m'
        return out

    # parse aggfunc
    if isinstance(aggfunc, str):
        aggfunc = getattr(np, aggfunc, None)
    if not callable(aggfunc):
        raise ValueError(
            'Parameter aggfunc must be a function or a name of numpy function.')

    total_seconds_arr = [str2seconds(x) for x in arr]
    agg_seconds = aggfunc(total_seconds_arr)
    out = seconds2str(agg_seconds)
    return out


def get_metrics_and_time(hist_df, *, primary_metric='f1_score', time='time',
                         metrics=['accuracy', 'top_3', 'f1_score']):
    assert primary_metric in hist_df
    metrics = [x for x in metrics if x in hist_df]
    assert len(metrics) > 0

    best_idx = hist_df[primary_metric].idxmax()
    out = hist_df.loc[best_idx, metrics].to_dict()
    if time in hist_df:
        out['mean_epoch_time'] = aggtime(hist_df[time], 'mean')
        out['total_time'] = aggtime(hist_df[time], 'sum')
    return out
    

def get_metrics_and_time_df(group_dict, *, primary_metric='f1_score', time='time',
                            metrics=['accuracy', 'top_3', 'f1_score']):
    """
    Return dataframe with metrics and time for each group.
    """
    out = {k: get_metrics_and_time(group_df, primary_metric=primary_metric,
                                   time=time, metrics=metrics)
           for k, group_df in group_dict.items()}
    out = pd.DataFrame.from_dict(out, orient='index')
    return out


def _compare_training_process(group_dict, xlim=[0, 30]):
    first_group_df = list(group_dict.values())[0]
    assert 'epoch' in first_group_df
    losses = [x for x in ['train_loss', 'valid_loss'] if x in first_group_df]
    metrics = [x for x in ['accuracy', 'f1_score'] if x in first_group_df]
    assert len(losses) > 0
    assert len(metrics) > 0

    ngroups = len(group_dict)
    fig, axs = viz.create_fig(ncols=ngroups, nrows=2, colsize=7, rowsize=5)

    for ax, (k, g) in zip(axs[:ngroups], group_dict.items()):
        params = dict(y=losses, xlim=xlim, ylim=[0.0, 6.0])
        viz.plot_training_progress(g, x='epoch', title=k, ax=ax, **params)
        ax.set(ylabel='Losses')

    for ax, (k, g) in zip(axs[ngroups:], group_dict.items()):
        params = dict(y=metrics, xlim=xlim, ylim=[0, 1])
        viz.plot_training_progress(g, x='epoch', title=k, ax=ax, **params)
        ax.set(ylabel='Metrics')

    plt.show()


def compare_training_process(group_dict, *, xlim=[0, 30], items_per_line=3):
    _groups = []
    for i, (k, v) in enumerate(group_dict.items()):
        if (i % items_per_line) == 0:
            _groups.append({})
        _groups[-1][k] = v

    for _group_dict in _groups:
        _compare_training_process(_group_dict, xlim)
