import torch

from ssfm_torch.losses import (
    UncertaintyDistillationLoss,
    UncertaintyJointLoss,
    UncertaintyMLP,
    UncertaintyScoreLoss,
)
from ssfm_torch.diffusion import VPDiffusion
from ssfm_torch.model import RMSGroupNorm, build_model


def tiny_model(dropout_rate=0.0):
    return build_model(
        resolution=8,
        base_channels=8,
        channel_mult=(1,),
        num_groups=1,
        dropout_rate=dropout_rate,
        attn_resolutions=(),
        head_dim=8,
        num_res_blocks=1,
        diff_base_channels=8,
        diff_channel_mult=(1,),
        diff_attn_resolutions=(),
    )


def test_rms_group_norm_uses_epsilon_outside_square_root():
    x = torch.arange(1, 1 + 2 * 4 * 4, dtype=torch.float32).reshape(1, 2, 4, 4)
    norm = RMSGroupNorm(1, eps=1e-4)
    expected = x / (x.square().mean().sqrt() + 1e-4)
    assert torch.allclose(norm(x), expected)


def test_model_batched_and_unbatched_outputs_match_in_eval_mode():
    torch.manual_seed(0)
    model = tiny_model().eval()
    x = torch.randn(3, 8, 8)
    W, H, K = (torch.randn_like(x) for _ in range(3))
    single = model(x, 0.9, 0.4, W, H, K)
    batched = model(
        x.unsqueeze(0), 0.9, 0.4, W.unsqueeze(0), H.unsqueeze(0), K.unsqueeze(0)
    )
    assert single.shape == x.shape
    assert torch.allclose(single, batched.squeeze(0), atol=2e-6, rtol=2e-5)


def test_joint_loss_backward_smoke():
    torch.manual_seed(1)
    model = tiny_model(dropout_rate=0.1)
    ema_model = tiny_model().eval().requires_grad_(False)
    ema_model.load_state_dict(model.state_dict())
    diffusion = VPDiffusion(1.0, 0.1, 20.0)
    uncertainty = UncertaintyMLP(8)
    loss_fn = UncertaintyJointLoss(
        UncertaintyScoreLoss(diffusion, dt=0.01),
        UncertaintyDistillationLoss(diffusion, dt=0.01, h_max=0.52),
        uncertainty,
        eta=0.5,
    )
    loss = loss_fn(model, ema_model, torch.randn(2, 3, 8, 8))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())

