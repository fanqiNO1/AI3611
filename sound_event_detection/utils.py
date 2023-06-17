import numpy as np
import pandas as pd
import scipy
import sklearn
from skmultilearn.model_selection import iterative_train_test_split


def encode_label(label, label_to_idx):
    target = np.zeros(len(label_to_idx))
    if isinstance(label, str):
        label = label.split(",")
    for l in label:
        target[label_to_idx[l]] = 1
    return target


def _train_test_split(data, test_size=0.1, y=None, is_stratified=True):
    if is_stratified:
        index_train, _, index_test, _ = iterative_train_test_split(
            data.index.values.reshape(-1, 1), y, test_size=test_size)
        train_data = data[data.index.isin(index_train.squeeze())]
        test_data = data[data.index.isin(index_test.squeeze())]
    else:
        train_data = data.sample(frac=1-test_size, random_state=10)
        test_data = data[~data.index.isin(train_data.index)]
    return train_data, test_data


def train_test_split(classinfo_filename, label_filename, test_size=0.1, is_stratified=True):
    label_to_idx, idx_to_label = {}, {}
    with open(classinfo_filename, 'r') as f:
        for line in f.readlines():
            idx, label = line.strip().split(",")
            label_to_idx[label] = int(idx)
            idx_to_label[int(idx)] = label
    label_data = pd.read_csv(label_filename, sep='\s+').convert_dtypes()
    label_data = label_data[label_data["event_labels"].notna()]
    label_array = label_data["event_labels"].apply(lambda x: encode_label(
        x, label_to_idx))
    label_array = np.stack(label_array.values)
    train_label, test_label = _train_test_split(
        label_data, test_size=test_size, y=label_array, is_stratified=is_stratified)
    return train_label, test_label, label_to_idx, idx_to_label


def find_contiguous_regions(activity_array):
    """
    Find contiguous regions from bool valued numpy.array.
    """
    # Find the changes in the activity_array
    change_indices = np.logical_xor(activity_array[1:],
                                    activity_array[:-1]).nonzero()[0]

    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, activity_array.size]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))


def _decode_timestamps(idx_to_label, labels):
    result_labels = []
    for i, label in enumerate(labels.T):
        row_indices = find_contiguous_regions(label)
        for row in row_indices:
            result_labels.append((idx_to_label[i], row[0], row[1]))
    return result_labels


def decode_timestamps(idx_to_label, labels):
    if labels.ndim == 3:
        return [_decode_timestamps(idx_to_label, label) for label in labels]
    else:
        return _decode_timestamps(idx_to_label, labels)


def binarize(pred, threshold=0.5):
    if pred.ndim == 3:
        return np.array([sklearn.preprocessing.binarize(sub, threshold=threshold) for sub in pred])
    else:
        return sklearn.preprocessing.binarize(pred, threshold=threshold)
    
    
def median_filter(x, window_size, threshold=0.5):
    x = binarize(x, threshold=threshold)
    if x.ndim == 3:
        size = (1, window_size, 1)
    elif x.ndim == 2 and x.shape[0] == 1:
        # Assume input is class-specific median filtering
        # E.g, Batch x Time  [1, 501]
        size = (1, window_size)
    elif x.ndim == 2 and x.shape[0] > 1:
        # Assume input is standard median pooling, class-independent
        # E.g., Time x Class [501, 10]
        size = (window_size, 1)
    return scipy.ndimage.median_filter(x, size=size)


def pred_to_time(df, ratio):
    df.onset = df.onset * ratio
    df.offset = df.offset * ratio
    return df
