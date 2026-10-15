"""Данные варианта 2: load_digits, как прямо указано в задании ЛР 1."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_variant2(layout='flat', center=True, bias=False):
    """Стратификация 60/20/20; среднее вычисляется только по train.

    flat: (N, 64), nchw: (N, 1, 8, 8), nhwc: (N, 8, 8, 1).
    Деление на 16 использует известную шкалу датчика, а не статистику test.
    """
    digits = load_digits()
    indices = np.arange(len(digits.target))
    trainval, test = train_test_split(indices, test_size=0.2, random_state=42,
                                      stratify=digits.target)
    train, val = train_test_split(trainval, test_size=0.25, random_state=42,
                                 stratify=digits.target[trainval])
    images = digits.data.astype(np.float64) / 16.0
    mean = images[train].mean(axis=0) if center else np.zeros(64)
    data = {'mean_image': mean.reshape(8, 8), 'image_shape': (8, 8)}
    for split, idx in [('train', train), ('val', val), ('test', test)]:
        x = images[idx] - mean
        if layout == 'nchw':
            x = x.reshape(-1, 1, 8, 8)
        elif layout == 'nhwc':
            x = x.reshape(-1, 8, 8, 1)
        elif layout != 'flat':
            raise ValueError('Unknown layout')
        if bias:
            if layout != 'flat':
                raise ValueError('Bias column requires flat input')
            x = np.column_stack([x, np.ones(len(x))])
        data['X_' + split], data['y_' + split] = x, digits.target[idx].copy()
        data['indices_' + split] = idx
    return data
