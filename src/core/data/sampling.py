import numpy as np

from torch.utils.data import WeightedRandomSampler

from .dataset import ImageDataset


__all__ = ['get_valid_col', 'create_dataset_sample', 'ClassUniformSampler']


def get_valid_col(train_df, label_col, valid_col, *, valid_pct=0.1):
    """Create Train/Validation data split."""
    def label_valid_col(group):
        if group.shape[0] >= 10:
            no_valid_items = int(np.floor(group.shape[0] * valid_pct))
        elif group.shape[0] >= 2:
            no_valid_items = 1
        else:
            no_valid_items = 0

        valid_mask = np.zeros(group.shape[0], dtype=bool)
        if no_valid_items > 0:
            valid_mask[-no_valid_items:] = True

        group[valid_col] = valid_mask
        return group

    train_df = train_df.copy()
    train_df[label_col] = train_df[label_col].fillna('')
    train_df = train_df.groupby(label_col).apply(label_valid_col)
    train_df[label_col] = train_df[label_col].replace('', np.nan)
    return train_df


def create_dataset_sample(train_df, label_col, max_items_per_class=1000):
    """Create dataset sample by selecting first `max_items_per_class` records in each class."""
    return (train_df.groupby(label_col)
            .apply(lambda g: g.head(max_items_per_class)).reset_index(drop=True))


def ClassUniformSampler(dataset: ImageDataset):
    label_col = dataset.label_col

    # create sampling weights
    class_counts = dataset.df[label_col].value_counts()
    desired_imgs_per_class = len(dataset) // len(class_counts)
    class_weight = (desired_imgs_per_class / class_counts).reset_index()
    class_weight.columns = [label_col, 'weight']
    sampling_weights = dataset.df.merge(
        class_weight, 'left', on=[label_col])['weight'].values
    assert len(sampling_weights) == len(dataset)

    # create weighted random sampler
    sampler = WeightedRandomSampler(
        sampling_weights, num_samples=len(dataset), replacement=True)

    return sampler
