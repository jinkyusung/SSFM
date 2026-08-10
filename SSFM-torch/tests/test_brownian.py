import math

import torch

from ssfm_torch.brownian import combine_levy, sample_levy


def test_levy_marginal_variances():
    generator = torch.Generator().manual_seed(7)
    count, dt = 100_000, -0.4
    levy = sample_levy(dt, (count,), generator=generator)
    assert abs(float(levy.W.var(correction=0)) - abs(dt)) < 0.01
    assert abs(float(levy.H.var(correction=0)) - abs(dt) / 12) < 0.002
    assert abs(float(levy.K.var(correction=0)) - abs(dt) / 720) < 4e-5


def test_chen_combination_has_correct_distribution_and_increment():
    generator = torch.Generator().manual_seed(11)
    count = 100_000
    first = sample_levy(-0.2, (count,), generator=generator)
    second = sample_levy(-0.3, (count,), generator=generator)
    whole = combine_levy(first, second)
    assert torch.equal(whole.W, first.W + second.W)
    assert torch.allclose(whole.dt, torch.tensor(-0.5))
    assert abs(float(whole.W.var(correction=0)) - 0.5) < 0.01
    assert abs(float(whole.H.var(correction=0)) - 0.5 / 12) < 0.002
    assert abs(float(whole.K.var(correction=0)) - 0.5 / 720) < 5e-5

