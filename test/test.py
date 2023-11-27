import unittest

import numpy as np
import torch

import discit.distr as distributions


RNG_SEED = 0
BATCH_SIZE = 256


class TestDistributions(unittest.TestCase):
    def test_multi_mixed(self):
        rng = np.random.default_rng(RNG_SEED)

        # Values
        values_1 = torch.tensor([
            [0., 0., 0., 0.],
            [1., 1., 1., 1.],
            [-1., -1., -1., -1.],
            [-1., 1., -1., 1.],
            [1., -1., 1., -1.]])

        values_2 = torch.tensor([[1.], [0.5]])

        # 5x4, 2x1 -> 4x5, 1x2 -> 4x5x1, 1x1x2 -> 4x5x2 -> 4x10 -> 10x4
        values = (values_1.T.unsqueeze(-1) * values_2.T.unsqueeze(-2)).flatten(-2).T
        n_cont_vars = 3

        # Raw inputs
        logits_1 = torch.from_numpy(rng.random((BATCH_SIZE, len(values_1)), dtype=np.float32))
        logits_2 = torch.from_numpy(rng.random((BATCH_SIZE, len(values_2)), dtype=np.float32))
        mean = torch.from_numpy(rng.random((BATCH_SIZE, n_cont_vars), dtype=np.float32))
        pseudo_log_dev = torch.from_numpy(rng.random((BATCH_SIZE, n_cont_vars), dtype=np.float32))

        # Distribution
        mcat_distr = distributions.MultiCategorical.from_raw(logits_1, logits_2, values=values)
        mnor_distr = distributions.MultiNormal.from_raw(mean, pseudo_log_dev, -np.log(3), 0.01, 3.)

        distr = distributions.MultiMixed(mcat_distr, mnor_distr)
        other_distr = distr

        # Callables
        sample_vals, sample_mnor_vals, sample_mcat_idcs = distr.sample()
        prob = distr.prob(sample_vals, sample_mnor_vals, sample_mcat_idcs)
        log_prob = distr.log_prob(sample_vals, sample_mnor_vals, sample_mcat_idcs)
        kl_div = distr.kl_div(other_distr)

        # Dims
        for i, attr in enumerate((distr.mcat.log_probs, distr.mcat.probs)):
            self.assertEqual(len(attr.shape), 2, i)
            self.assertEqual(attr.shape[0], BATCH_SIZE, i)
            self.assertEqual(attr.shape[1], values.shape[0], i)

        for i, attr in enumerate((distr.mean, distr.mode, distr.log_dev, distr.dev, distr.var, sample_vals)):
            self.assertEqual(len(attr.shape), 2, i)
            self.assertEqual(attr.shape[0], BATCH_SIZE, i)
            self.assertEqual(attr.shape[1], values.shape[1] + n_cont_vars, i)

        for i, attr in enumerate((distr.entropy, prob, log_prob, kl_div)):
            self.assertEqual(len(attr.shape), 1, i)
            self.assertEqual(len(attr), BATCH_SIZE, i)

        self.assertEqual(len(sample_mcat_idcs.shape), 2)
        self.assertEqual(sample_mcat_idcs.shape[0], BATCH_SIZE)
        self.assertEqual(sample_mcat_idcs.shape[1], 1)

    def test_multi_categorical(self):
        rng = np.random.default_rng(RNG_SEED)

        # Values
        values_1 = torch.tensor([
            [0., 0., 0., 0.],
            [1., 1., 1., 1.],
            [-1., -1., -1., -1.],
            [-1., 1., -1., 1.],
            [1., -1., 1., -1.]])

        values_2 = torch.tensor([[1.], [0.5]])

        # 5x4, 2x1 -> 4x5, 1x2 -> 4x5x1, 1x1x2 -> 4x5x2 -> 4x10 -> 10x4
        values = (values_1.T.unsqueeze(-1) * values_2.T.unsqueeze(-2)).flatten(-2).T

        # Raw inputs
        logits_1 = torch.from_numpy(rng.random((BATCH_SIZE, len(values_1)), dtype=np.float32))
        logits_2 = torch.from_numpy(rng.random((BATCH_SIZE, len(values_2)), dtype=np.float32))

        # Distribution
        distr = distributions.MultiCategorical.from_raw(logits_1, logits_2, values=values)
        other_distr = distributions.MultiCategorical(values, distr.log_probs, distr.probs)

        # Callables
        sample_vals, sample_idcs = distr.sample()
        prob = distr.prob(sample_vals, sample_idcs)
        log_prob = distr.log_prob(sample_vals, sample_idcs)
        kl_div = distr.kl_div(other_distr)

        # Dims
        for i, attr in enumerate((distr.log_probs, distr.probs)):
            self.assertEqual(len(attr.shape), 2, i)
            self.assertEqual(attr.shape[0], BATCH_SIZE, i)
            self.assertEqual(attr.shape[1], values.shape[0], i)

        for i, attr in enumerate((distr.mean, distr.mode, distr.log_dev, distr.dev, distr.var, sample_vals)):
            self.assertEqual(len(attr.shape), 2, i)
            self.assertEqual(attr.shape[0], BATCH_SIZE, i)
            self.assertEqual(attr.shape[1], values.shape[1], i)

        for i, attr in enumerate((distr.entropy, prob, log_prob, kl_div)):
            self.assertEqual(len(attr.shape), 1, i)
            self.assertEqual(len(attr), BATCH_SIZE, i)

        self.assertEqual(len(sample_idcs.shape), 2)
        self.assertEqual(sample_idcs.shape[0], BATCH_SIZE)
        self.assertEqual(sample_idcs.shape[1], 1)

    def test_inter_categorical(self):
        rng = np.random.default_rng(RNG_SEED)

        # Values
        values = torch.tensor([0.] + [2.**i for i in range(8)]).unsqueeze(-1)

        # Raw inputs
        logits = torch.from_numpy(rng.random((BATCH_SIZE, len(values)), dtype=np.float32))

        # Distribution
        distr = distributions.InterCategorical.from_raw(logits, values=values)

        # Callables
        sample_vals, sample_idcs = distr.sample()
        prob = distr.prob(sample_vals, sample_idcs)
        log_prob = distr.log_prob(sample_vals, sample_idcs)

        # Dims
        for i, attr in enumerate((prob, log_prob)):
            self.assertEqual(len(attr.shape), 1, i)
            self.assertEqual(len(attr), BATCH_SIZE, i)

    def test_multi_normal(self):
        rng = np.random.default_rng(RNG_SEED)

        # Values
        n_cont_vars = 3

        # Raw inputs
        mean = torch.from_numpy(rng.random((BATCH_SIZE, n_cont_vars), dtype=np.float32))
        log_dev = torch.from_numpy(rng.random((BATCH_SIZE, n_cont_vars), dtype=np.float32))

        # Distribution
        distr = distributions.MultiNormal.from_raw(mean, log_dev, -np.log(3), 0.01, 3.)
        other_distr = distributions.MultiNormal(distr.mean, distr.log_dev, distr.dev)

        # Callables
        sample_vals, = distr.sample()
        prob = distr.prob(sample_vals)
        log_prob = distr.log_prob(sample_vals)
        kl_div = distr.kl_div(other_distr)

        # Dims
        for i, attr in enumerate((distr.mean, distr.mode, distr.log_dev, distr.dev, distr.var, sample_vals)):
            self.assertEqual(len(attr.shape), 2, i)
            self.assertEqual(attr.shape[0], BATCH_SIZE, i)
            self.assertEqual(attr.shape[1], n_cont_vars, i)

        for i, attr in enumerate((distr.entropy, prob, log_prob, kl_div)):
            self.assertEqual(len(attr.shape), 1, i)
            self.assertEqual(len(attr), BATCH_SIZE, i)

    def test_fixed_var_normal(self):
        rng = np.random.default_rng(RNG_SEED)

        # Raw inputs
        mean_1 = torch.from_numpy(rng.random((BATCH_SIZE, 1), dtype=np.float32))
        mean_2 = torch.from_numpy(rng.random((BATCH_SIZE, 1), dtype=np.float32))

        # Distributions
        distr = distributions.FixedVarNormal(mean_1)
        other_distr = distributions.FixedVarNormal(mean_2)

        # Callables
        sample_vals, = distr.sample()
        prob = distr.prob(sample_vals)
        log_prob = distr.log_prob(sample_vals)
        kl_div = distr.kl_div(other_distr)

        # Dims
        for i, attr in enumerate((distr.mean, distr.mode, distr.log_dev, distr.dev, distr.var, sample_vals)):
            self.assertEqual(len(attr.shape), 2, i)
            self.assertEqual(attr.shape[0], BATCH_SIZE, i)
            self.assertEqual(attr.shape[1], 1, i)

        for i, attr in enumerate((distr.entropy, prob, log_prob, kl_div)):
            self.assertEqual(len(attr.shape), 1, i)
            self.assertEqual(len(attr), BATCH_SIZE, i)


if __name__ == '__main__':
    unittest.main()
