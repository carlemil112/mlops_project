import torch
from torch import nn, optim


def test_checkpoint_roundtrip(
    tmp_path,
):  # tmp path is a midlertidig folder, which only exists under test
    import model.patch_train as pt

    # Minimal generator/discriminator
    gen = nn.Conv2d(1, 3, kernel_size=1)
    disc = nn.Conv2d(4, 1, kernel_size=1)

    opt_g = optim.Adam(gen.parameters(), lr=1e-3)
    opt_d = optim.Adam(disc.parameters(), lr=1e-3)
    sched = optim.lr_scheduler.StepLR(opt_g, step_size=1, gamma=0.1)

    ckpt_path = tmp_path / "ckpt.tar"

    pt.create_checkpoint.__wrapped__(
        epoch=7,
        model=gen,
        disc=disc,
        optimizer_G=opt_g,
        optimizer_D=opt_d,
        scheduler=sched,
        loss=1.23,
        val_loss=2.34,
        run_id="abc123",
        path=str(ckpt_path),
    )

    # Reset weights to see if load is doing anything
    for p in gen.parameters():
        nn.init.constant_(p, 0.0)

    gen2, disc2, opt_g2, opt_d2, sched2, epoch, loss, val_loss, run_id = (
        pt.load_checkpoint(
            path=str(ckpt_path),
            model=gen,
            disc=disc,
            optimizer_G=opt_g,
            optimizer_D=opt_d,
            scheduler=sched,
        )
    )

    assert epoch == 7
    assert loss == 1.23
    assert val_loss == 2.34
    assert run_id == "abc123"

    # check at generator weights ikke længere er alle 0 efter load
    w = next(gen2.parameters()).detach()
    assert torch.any(w != 0.0)
