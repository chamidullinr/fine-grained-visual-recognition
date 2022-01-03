import os

import numpy as np

from fastprogress.fastprogress import master_bar, progress_bar


class BaseProgress:
    def __init__(self):
        self.score_names = None

    def _write_table_stats(self, row_items):
        return NotImplementedError

    def _format_row_items(self, row_items):
        return [f'{x:.6f}' if isinstance(x, (float, np.floating)) else str(x)
                for x in row_items]

    def log_epoch_scores(self, scores):
        if self.score_names is None:
            self.score_names = list(scores.keys())
            col_items = self._format_row_items(self.score_names)
            self._write_table_stats(col_items)
        row_items = self._format_row_items(scores.values())
        self._write_table_stats(row_items)


class ProgressBar(BaseProgress):
    def __init__(self):
        super().__init__()
        self.mbar = None
        self.pbar = None

    def master_bar(self, iterable, total=None):
        self.mbar = master_bar(iterable, total=total)
        return self.mbar

    def progress_bar(self, iterable, total=None):
        self.pbar = progress_bar(iterable, total=total, parent=self.mbar)
        self.pbar.update(0)
        return self.pbar

    def _write_table_stats(self, row_items):
        if self.mbar is not None:
            self.mbar.write(row_items, table=True)


class CSVProgress(BaseProgress):
    def __init__(self, filename='history.csv', path='.'):
        super().__init__()
        self.filename = filename
        self.path = path
        self.filepath = os.path.join(self.path, self.filename)
        # io.create_path(self.path)

    def _write_table_stats(self, row_items):
        with open(self.filepath, 'a') as f:
            f.write(','.join(row_items) + '\n')


def format_seconds(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        out = f'{h:d}:{m:02d}:{s:02d}'
    else:
        out = f'{m:02d}:{s:02d}'
    return out
