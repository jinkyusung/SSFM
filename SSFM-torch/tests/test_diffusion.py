import torch

from ssfm_torch.diffusion import VPDiffusion


def test_vp_diffusion_closed_form_and_batch_broadcasting():
    diffusion = VPDiffusion(reverse_eta=1.0, beta_min=0.1, beta_max=20.0)
    time = torch.tensor([0.2, 0.7])
    y0 = torch.ones(2, 3, 4, 4)
    noise = torch.full_like(y0, 0.25)
    sample, score = diffusion.forward_sample(time, y0, noise=noise)

    alpha = torch.exp(-0.5 * (0.1 * time + 0.5 * 19.9 * time.square()))
    sigma = torch.sqrt(1 - alpha.square())
    expected = alpha[:, None, None, None] + sigma[:, None, None, None] * noise
    assert torch.allclose(sample, expected)
    assert torch.allclose(score, -noise / sigma[:, None, None, None])
    assert diffusion.reverse_drift(time, sample, score).shape == y0.shape

