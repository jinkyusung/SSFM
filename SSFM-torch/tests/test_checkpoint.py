import copy

import torch

from ssfm_torch.checkpoint import load_checkpoint, save_checkpoint
from ssfm_torch.losses import UncertaintyMLP
from ssfm_torch.model import build_model


def tiny_model():
    return build_model(
        resolution=8,
        base_channels=8,
        channel_mult=(1,),
        num_groups=1,
        dropout_rate=0.0,
        attn_resolutions=(),
        head_dim=8,
        num_res_blocks=1,
        diff_base_channels=8,
        diff_channel_mult=(1,),
        diff_attn_resolutions=(),
    )


def test_checkpoint_roundtrip(tmp_path):
    model = tiny_model()
    ema_model = copy.deepcopy(model)
    uncertainty = UncertaintyMLP(8)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(uncertainty.parameters()), lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    save_checkpoint(
        str(tmp_path), model, ema_model, uncertainty, optimizer, scheduler, 17, {"x": 1}
    )

    expected = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    state = load_checkpoint(
        str(tmp_path), model, ema_model, uncertainty, optimizer, scheduler
    )
    assert state["step"] == 17
    assert state["config"] == {"x": 1}
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name])
