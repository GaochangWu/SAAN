import h5py
import numpy as np


def read_data(file, Flag=1):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        train_data = np.transpose(data, (0, 2, 4, 3, 1))
        train_label = np.transpose(label, (0, 2, 4, 3, 1))

        if train_data.shape[4] == 3 and Flag:
            train_data = 0.2989 * train_data[:, :, :, :, 0:1] + 0.587 * train_data[:, :, :, :, 1:2] \
                         + 0.114 * train_data[:, :, :, :, 2:3]
        if train_label.shape[4] == 3 and Flag:
            train_label = 0.2989 * train_label[:, :, :, :, 0:1] + 0.587 * train_label[:, :, :, :, 1:2] \
                         + 0.114 * train_label[:, :, :, :, 2:3]
        return train_data, train_label


def read_data2D(file, Flag=1):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        train_data = np.transpose(data, (0, 3, 2, 1))
        train_label = np.transpose(label, (0, 3, 2, 1))

        if train_data.shape[4] == 3 and Flag:
            train_data = 0.2989 * train_data[:, :, :, 0:1] + 0.587 * train_data[:, :, :, 1:2] \
                         + 0.114 * train_data[:, :, :, 2:3]
        return train_data, train_label
