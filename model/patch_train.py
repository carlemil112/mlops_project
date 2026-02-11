import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.unet import Unet as unet
from models.discriminator import Discriminator
from data_loader import gray_color_data
import os
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")


def create_checkpoint(epoch, model, disc, optimizer_G, optimizer_D, scheduler, loss, val_loss, run_id, path):
    """ creating a checkpoint """
    torch.save(
        {
            'epoch': epoch,
            'gen_state_dict': model.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            "loss": loss,
            "val_loss": val_loss,
            "run_id": run_id,
        },
        path
    )


def load_checkpoint(path, model, disc, optimizer_G, optimizer_D, scheduler):
    """ loading a checkpoint """
    checkpoint = torch.load(path, weights_only=False)

    def strip_module(sd):
        # if keys start with "module.", remove that prefix
        from collections import OrderedDict
        new_sd = OrderedDict()
        for k, v in sd.items():
            new_k = k.replace("module.", "")
            new_sd[new_k] = v
        return new_sd

    # --- load generator ---
    gen_sd = checkpoint["gen_state_dict"]
    try:
        model.load_state_dict(gen_sd)
    except RuntimeError:
        # probably saved with DataParallel → strip "module."
        model.load_state_dict(strip_module(gen_sd))

    # --- load discriminator ---
    disc_sd = checkpoint["disc_state_dict"]
    try:
        disc.load_state_dict(disc_sd)
    except RuntimeError:
        disc.load_state_dict(strip_module(disc_sd))

    # optimizers + scheduler
    optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
    optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    val_loss = checkpoint["val_loss"]
    run_id = checkpoint["run_id"]

    return model, disc, optimizer_G, optimizer_D, scheduler, epoch, loss, val_loss, run_id



# Training method for discriminator and generator. Setup is based on pix2pix.
# 
def train(model, disc, device, optimizer_G, optimizer_D, train_loader, epoch):
    """ training loop with unet and patchgan """
    model.train()  # unet generator
    disc.train()   # patchGAN discriminator

    # adversarial loss for patchgan (outputs with sigmoid in discriminator)
    adv_criterion = nn.BCELoss()

    # reconstruction loss with l1 for holding the structure of the image
    rec_criterion = nn.L1Loss()

    # weight for L1
    lambda_l1 = 100.0

    last_g_loss = 0.0

    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)  # gs input
        y = y.to(device)  # color target

        # !!! DISCRIMINATOR TRAINING !!!

        # generate fake color image
        with torch.no_grad():
            fake_y = model(x)

        # concatenate grayscale and color along channel dim
        # real pics (gray and color)
        real_pair = torch.cat([x, y], dim=1)   # B, 4, H, W
        # fake pics (gray and fake color)
        fake_pair = torch.cat([x, fake_y], dim=1)  # B, 4, H, W

        optimizer_D.zero_grad()

        # classify real as 1
        pred_real = disc(real_pair)
        real_labels = torch.ones_like(pred_real, device=device)
        d_loss_real = adv_criterion(pred_real, real_labels)

        # and fake with 0
        pred_fake = disc(fake_pair)
        fake_labels = torch.zeros_like(pred_fake, device=device)
        d_loss_fake = adv_criterion(pred_fake, fake_labels)

        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_loss.backward()
        optimizer_D.step()

        # !!! GENERATOR TRAINING !!!

        optimizer_G.zero_grad()
        # generate fake images with generator
        fake_y = model(x)
        fake_pair_for_g = torch.cat([x, fake_y], dim=1)

        # generator tries to make discriminator think that their fake pics are real
        pred_fake_for_g = disc(fake_pair_for_g)
        real_labels_for_g = torch.ones_like(pred_fake_for_g)
        g_adv_loss = adv_criterion(pred_fake_for_g, real_labels_for_g)

        # l1 reconstruction loss between fake color and real color
        raw_l1 = rec_criterion(fake_y, y)
        g_l1_loss = raw_l1 * lambda_l1

        # sum of generator loss
        g_loss = g_adv_loss + g_l1_loss
        g_loss.backward()
        optimizer_G.step()

        raw_l1 = rec_criterion(fake_y, y)
        last_g_loss = g_loss.item()

        if i % 200 == 0:
            print(
                "Train epoch: %s , iteration: %s , G loss: %.4f , D loss: %.4f"
                % (epoch, i, g_loss.item(), d_loss.item())
            )
            wandb.log({
                "train G total loss": g_loss.item(),
                "train G adv loss(how much d catches fake)": g_adv_loss.item(),
                "train G L1 loss (weighted)": g_l1_loss.item(),
                "train G L1 loss raw": raw_l1.item(),
                "train D loss": d_loss.item()
            })

    return last_g_loss


def val(model, device, val_loader):
    """ validation loop """
    model.eval()
    lossfunction = nn.L1Loss()
    total_loss = 0
    total_batch_size = 0

    with torch.no_grad():
        loss = 0
        for batch, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            batch_loss = lossfunction(y, output).item()
            batch_size = x.size(0)
            total_loss += batch_loss * batch_size
            total_batch_size += batch_size

        # (originally blank line 148)
        loss = total_loss / total_batch_size

    print("\nVal L1: ", loss, "\n")
    wandb.log({"Val L1": loss})
    return loss


def save_test_image(pred_tensor, target_tensor, epoch):
    out_dir = "test_output_images/rasmus_places_cropped"
    os.makedirs(out_dir, exist_ok=True)  # ensure folder exists

    pred_img = transforms.functional.to_pil_image(
        pred_tensor[0].detach().cpu()
    )
    pred_img.save(f"{out_dir}/pred_img{epoch}.jpg")

    target_img = transforms.functional.to_pil_image(
        target_tensor[0].detach().cpu()
    )
    target_img.save(f"{out_dir}/target_img{epoch}.jpg")


def main(cfg: DictConfig):
    # setting up variables
    number_epochs = cfg.number_epochs
    batch_size = cfg.train_batch_size
    learning_rate = cfg.learning_rate
    torch.manual_seed(cfg.seed)  # secure reproducibility
    epoch = cfg.checkpoint_epoch # Checkpoint
    loss = 0 # What do we even use this for
    val_loss = cfg.initial_loss_value

    # start up wandb
    wandb.login()

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda")
    else:
        device = torch.device("cpu")
        print("cpu")

    # download of data
    data_path = "data/"
    checkpoint_path = "checkpoint/patch_unet.tar"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # download data
    train_set = gray_color_data(data_path, split="train-standard", train=True)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=16
    )

    val_set = gray_color_data(data_path, split="val", train=False)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=16
    )

    # initilizing model
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(unet())
        disc = nn.DataParallel(Discriminator(input_channels=4))
        print(f"Using {torch.cuda.device_count()} GPU'S")
    else:
        model = unet()
        disc = Discriminator(input_channels=4)
        print(f"Using {torch.cuda.device_count()} GPU'S")

    model = model.to(device)
    disc = disc.to(device)

    # define optimizers
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=cfg.weight_decay)
    # these betas are good for a discriminator optimizer
    optimizer_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=cfg.adam_betas)

    scheduler = StepLR(optimizer, step_size=cfg.lr_scheduler_step_size, gamma=0.5)  # only for gen


    # Checkpoint loader. If checkpoint file is present, continue training from checkpoint file.
    if os.path.exists(checkpoint_path):
        (model, disc, optimizer, optimizer_disc, scheduler, epoch, loss, val_loss, run_id) = load_checkpoint(
            checkpoint_path, model, disc, optimizer, optimizer_disc, scheduler
        )
        run = wandb.init(project="unet_places_cropped", id=run_id, resume="must")
        epoch += 1
        print("Checkpoints is loaded")
    else:
        run = wandb.init(project="unet_places_cropped", name="unet-places365-cropped")
        run_id = run.id

    for epoch in range(epoch, number_epochs):
        current_lr = scheduler.get_last_lr()[0]
        print("LR", current_lr)
        print("batch size", batch_size)

        train_loss = train(model, disc, device, optimizer, optimizer_disc, train_loader, epoch)
        new_val_loss = val(model, device, val_loader)

        wandb.log({"Epoch": epoch})
        wandb.log({"Learning_rate": current_lr})

        scheduler.step()

        if new_val_loss < val_loss:
            print("Lower val result")
            create_checkpoint(
                epoch,
                model,
                disc,
                optimizer,
                optimizer_disc,
                scheduler,
                train_loss,
                new_val_loss,
                run_id,
                checkpoint_path
            )
            print("New checkpoint has been made")
            val_loss = new_val_loss  # this wasnt in the original, we need to update this?

        # Save a recreated image to show progress ever x epoch.
        if epoch % 2 == 0:
            for x, y in val_loader:
                test_img = x.to(device)
                model.eval()
                test_pred = model(test_img)
                save_test_image(test_pred, y, epoch)
                break

    torch.save({"gen": model.state_dict(), "disc": disc.state_dict()}, "rasmus_cropped_unet_adam.pt")

if __name__ == "__main__":
    main()
