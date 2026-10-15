"""Численные проверки производных и граничных случаев: python tools/check_implementations.py."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'lab_1-2'))
from digits_data import load_variant2
from scripts.classifiers.k_nearest_neighbor import KNearestNeighbor
from scripts.classifiers.linear_svm import svm_loss_naive, svm_loss_vectorized
from scripts.classifiers.softmax import softmax_loss_naive, softmax_loss_vectorized
from scripts.classifiers.neural_net import TwoLayerNet as StandaloneNet
from scripts.classifiers.linear_classifier import LinearSVM
# The course ships two independent packages both named scripts.
for name in list(sys.modules):
    if name == 'scripts' or name.startswith('scripts.'):
        del sys.modules[name]
sys.path[0] = str(ROOT / 'lab_3')
from scripts.layers import *
from scripts.fast_layers import (conv_forward_fast, conv_backward_fast,
    conv_forward_im2col, conv_backward_im2col, max_pool_forward_fast, max_pool_backward_fast)
from scripts.classifiers.fc_net import FullyConnectedNet, TwoLayerNet
from scripts.classifiers.cnn import ThreeLayerConvNet
from scripts import optim
from scripts.solver import Solver


def numeric(fn, x, upstream=None, h=1e-5):
    result = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        old = x[index]
        x[index] = old + h
        plus = np.array(fn(), copy=True)
        x[index] = old - h
        minus = np.array(fn(), copy=True)
        x[index] = old
        diff = (plus - minus) / (2 * h)
        result[index] = np.sum(diff if upstream is None else diff * upstream)
    return result


class Implementations(unittest.TestCase):
    def setUp(self):
        np.random.seed(10)

    def check_grad(self, fn, x, actual, upstream=None):
        np.testing.assert_allclose(numeric(fn, x, upstream), actual, atol=2e-7, rtol=2e-4)

    def test_split_and_training_only_center(self):
        data = load_variant2()
        groups = [set(data['indices_' + split]) for split in ['train', 'val', 'test']]
        self.assertEqual(len(set.union(*groups)), 1797)
        for i in range(3):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])
        np.testing.assert_allclose(data['X_train'].mean(axis=0), 0, atol=1e-14)
        for layout, shape in [('flat', (64,)), ('nchw', (1, 8, 8)), ('nhwc', (8, 8, 1))]:
            self.assertEqual(load_variant2(layout)['X_train'].shape[1:], shape)

    def test_knn_distances_ties_and_k(self):
        model = KNearestNeighbor()
        x = np.random.randn(7, 5); y = np.array([0, 1, 2, 0, 1, 2, 2])
        model.train(x, y)
        for fn in [model.compute_distances_two_loops, model.compute_distances_one_loop]:
            np.testing.assert_allclose(fn(x), model.compute_distances_no_loops(x), atol=1e-7)
        self.assertEqual(model.predict_labels(np.array([[1., 2., 3., 4., 5., 6., 7.]]), 2)[0], 0)
        with self.assertRaises(ValueError): model.predict(x, k=8)

    def test_linear_losses_and_gradients(self):
        x = np.random.randn(5, 4); y = np.array([0, 1, 2, 1, 0]); w = 0.02 * np.random.randn(4, 3)
        for naive, vector in [(svm_loss_naive, svm_loss_vectorized), (softmax_loss_naive, softmax_loss_vectorized)]:
            for reg in [0., 0.1]:
                a, da = naive(w, x, y, reg); b, db = vector(w, x, y, reg)
                np.testing.assert_allclose(a, b)
                np.testing.assert_allclose(da, db, atol=1e-12)
                self.check_grad(lambda: vector(w, x, y, reg)[0], w, db)
        loss, grad = softmax_loss_vectorized(w * 1e6, x, y, 0)
        self.assertTrue(np.isfinite(loss) and np.isfinite(grad).all())
        classifier = LinearSVM()
        with self.assertRaises(ValueError): classifier.predict(x)

    def test_network_gradients(self):
        x = np.random.randn(4, 5); y = np.array([0, 1, 2, 0])
        models = [StandaloneNet(5, 6, 3, std=0.1), TwoLayerNet(5, 6, 3, 0.1, 0.1)]
        for norm in [None, 'batchnorm', 'layernorm']:
            models.append(FullyConnectedNet([6, 4], input_dim=5, num_classes=3,
                normalization=norm, dropout=0.75, seed=17, weight_scale=0.1,
                reg=0.1, dtype=np.float64))
        for model in models:
            fn = lambda: model.loss(x, y, reg=0.1) if isinstance(model, StandaloneNet) else model.loss(x, y)
            _, grads = fn()
            for key, param in model.params.items():
                with self.subTest(model=type(model).__name__, param=key):
                    self.check_grad(lambda: fn()[0], param, grads[key])

    def test_norm_and_dropout(self):
        x = np.random.randn(3, 4); g = np.random.randn(4); b = np.random.randn(4)
        dy = np.random.randn(3, 4)
        out, cache = batchnorm_forward(x, g, b, {'mode': 'train'})
        dx, dg, db = batchnorm_backward(dy, cache)
        for a, alt in zip((dx, dg, db), batchnorm_backward_alt(dy, cache)):
            np.testing.assert_allclose(a, alt, atol=1e-12)
        self.check_grad(lambda: batchnorm_forward(x, g, b, {'mode': 'train'})[0], x, dx, dy)
        params = {'mode': 'train', 'p': 0.75, 'seed': 10}
        out, cache = dropout_forward(x, params)
        params['mode'] = 'test'
        # Cache must keep the mode from the corresponding forward pass.
        self.check_grad(lambda: dropout_forward(x, {'mode': 'train', 'p': .75, 'seed': 10})[0],
                        x, dropout_backward(dy, cache), dy)
        np.testing.assert_array_equal(dropout_forward(x, {'mode': 'test', 'p': .75})[0], x)

    def test_convolution_padding_stride_and_rectangles(self):
        for pad, stride, shape, kernel in [(0, 1, (2, 2, 4, 5), (2, 2, 2, 3)),
                                            (1, 2, (2, 1, 5, 7), (2, 1, 3, 3))]:
            x = np.random.randn(*shape); w = np.random.randn(*kernel); b = np.random.randn(2)
            cp = {'pad': pad, 'stride': stride}
            out, cache = conv_forward_naive(x, w, b, cp); dy = np.random.randn(*out.shape)
            grads = conv_backward_naive(dy, cache)
            for param, actual in zip((x, w, b), grads):
                self.check_grad(lambda: conv_forward_naive(x, w, b, cp)[0], param, actual, dy)
            for forward, backward in [(conv_forward_fast, conv_backward_fast),
                                       (conv_forward_im2col, conv_backward_im2col)]:
                fast, fc = forward(x, w, b, cp)
                np.testing.assert_allclose(out, fast, atol=1e-12)
                for a, other in zip(grads, backward(dy, fc)):
                    np.testing.assert_allclose(a, other, atol=1e-12)

    def test_pool_overlapping_and_ties(self):
        x = np.random.randn(2, 2, 4, 5)
        pp = {'pool_height': 2, 'pool_width': 3, 'stride': 1}
        out, cache = max_pool_forward_naive(x, pp); dy = np.random.randn(*out.shape)
        dx = max_pool_backward_naive(dy, cache)
        self.check_grad(lambda: max_pool_forward_naive(x, pp)[0], x, dx, dy)
        other, oc = max_pool_forward_fast(x, pp)
        np.testing.assert_allclose(out, other)
        np.testing.assert_allclose(dx, max_pool_backward_fast(dy, oc))
        out, cache = max_pool_forward_naive(np.ones((1, 1, 2, 2)),
            {'pool_height': 2, 'pool_width': 2, 'stride': 2})
        np.testing.assert_allclose(max_pool_backward_naive(np.ones_like(out), cache), .25)

    def test_cnn_gradients(self):
        x = np.random.randn(2, 1, 4, 4); y = np.array([0, 2])
        model = ThreeLayerConvNet(input_dim=(1, 4, 4), num_filters=2, filter_size=3,
            hidden_dim=3, num_classes=3, weight_scale=.1, reg=.1, dtype=np.float64)
        _, grads = model.loss(x, y)
        for key, param in model.params.items():
            self.check_grad(lambda: model.loss(x, y)[0], param, grads[key])

    def test_optimizer_steps(self):
        w = np.array([1., -2.]); grad = np.array([.1, -.2])
        a, _ = optim.sgd(w.copy(), grad, {'learning_rate': .1})
        b, _ = optim.sgd_momentum(w.copy(), grad, {'learning_rate': .1, 'momentum': 0})
        np.testing.assert_allclose(a, b)
        a, state = optim.adam(w, grad, {'learning_rate': .01})
        np.testing.assert_allclose(a, w - .01 * grad / (np.abs(grad) + 1e-8))
        self.assertEqual(state['t'], 1)
        a, state = optim.adam(a, grad, state)
        self.assertEqual(state['t'], 2)

    def test_solver_restores_batchnorm_state(self):
        class StatefulModel:
            def __init__(self):
                self.params = {'W': np.array([0.])}
                self.bn_params = [{'running_mean': np.array([0.])}]
            def loss(self, x, y=None):
                if y is not None:
                    self.bn_params[0]['running_mean'] += 1
                    return 1., {'W': np.array([-1.])}
                # Accuracy peaks at the first update; later checkpoints are worse.
                good = self.params['W'][0] < .015
                return np.tile([1., 0.] if good else [0., 1.], (len(x), 1))
        data = {'X_train': np.ones((2, 1)), 'y_train': np.zeros(2, dtype=int),
                'X_val': np.ones((2, 1)), 'y_val': np.zeros(2, dtype=int)}
        model = StatefulModel(); solver = Solver(model, data, batch_size=2, num_epochs=3, verbose=False)
        solver.train()
        np.testing.assert_allclose(model.params['W'], .01)
        np.testing.assert_allclose(model.bn_params[0]['running_mean'], 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
