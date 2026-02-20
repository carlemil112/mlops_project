import torch
from torch import nn
from torch import optim
from torchvision import transforms
from data_loader import gray_color_data
import os

import wandb

# MLFlow import
import mlflow


def create_checkpoint(epoch, model, optimizer, loss, val_loss, run_id, path):
    """creating a checkpoint"""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "val_loss": val_loss,
            "run_id": run_id,
        },
        path,
    )


def load_checkpoint(path, model, optimizer):
    """loading a checkpoint"""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    val_loss = checkpoint["val_loss"]
    run_id = checkpoint["run_id"]

    return model, optimizer, epoch, loss, val_loss, run_id


def train(model, device, optimizer, train_loader, epoch):
    """training loop"""
    print("training")
    model.train()
    loss_function = nn.MSELoss()
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)
        loss = loss_function(y, y_pred)
        loss.backward()
        optimizer.step()
        print("training")
        if i % 200 == 0:
            print(
                "Train Epoch: %s , Iteration: %s , Train Loss: %s"
                % (epoch, i, loss.item())
            )
            wandb.log({"Train loss": loss.item()})
            # MLFlow loss tracking
            mlflow.log_metric("train/mse", loss.item(), step=epoch)

    return loss.item


def val(model, device, val_loader):
    """validation loop"""
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

        loss = total_loss / total_batch_size
        print("\nVal MAE: ", loss, "\n")
        wandb.log({"Val mae": loss})
        mlflow.log_metric("val/mae", loss)

    return total_loss


def save_test_image(pred_tensor, target_tensor, epoch):
    pred_img = transforms.functional.to_pil_image(pred_tensor[0].detach().cpu())
    pred_img.save(f"test_output_images/pretrain_pred_img{epoch}.jpg")
    target_img = transforms.functional.to_pil_image(target_tensor[0].detach().cpu())
    target_img.save(f"test_output_images/pretrain_target_img{epoch}.jpg")


def main():
    # setting up variables
    number_epochs = 100
    batch_size = 64
    learning_rate = 1e-3
    torch.manual_seed(42)  # secure reproducibility
    epoch = 1
    loss = 0
    val_loss = 100000000

    # MLFlow hyperparameters tracking
    mlflow.set_experiment("recolorization-gan")  # :contentReference[oaicite:4]{index=4}
    with mlflow.start_run(run_name="unet_pretrain_train"):
        mlflow.log_params(
            {
                "number_epochs": number_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": 42,
                "checkpoint_path": "/checkpoint/pretrained_unet.tar",
                "data_path": "/data",
            }
        )

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
    data_path = "/data"
    checkpoint_path = "/checkpoint/pretrained_unet.tar"

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
    model = torch.hub.load(
        "milesial/Pytorch-UNet", "unet_carvana", pretrained=True, scale=1
    )
    model.inc.double_conv[0] = nn.Conv2d(
        1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.outc.conv = nn.Conv2d(
        64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    print(model)

    model = model.to(device)

    # define optimizers
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001
    )

    if os.path.exists(checkpoint_path):
        model, optimizer, epoch, loss, val_loss, run_id = load_checkpoint(
            checkpoint_path, model, optimizer
        )
        run = wandb.init(project="unet_places_cropped", id=run_id, resume="must")
        epoch += 1
        print("Checkpoints is loaded")
    else:
        run = wandb.init(project="unet_places_cropped", name="cropped-image_pretrained")
        run_id = run.id

    for epoch in range(epoch, number_epochs):
        train_loss = train(model, device, optimizer, train_loader, epoch)
        new_val_loss = val(model, device, val_loader)

        if new_val_loss < val_loss:
            print("Lower val result")
            create_checkpoint(
                epoch,
                model,
                optimizer,
                train_loss,
                new_val_loss,
                run_id,
                checkpoint_path,
            )
            print("New checkpoint has been made")

        # saving one image to check progress
        if epoch % 10 == 0:
            for x, y in val_loader:
                test_img = x.to(device)
                model.eval()
                test_pred = model(test_img)
                save_test_image(test_pred, y, epoch)
                break

        # MLFlow save checkpoints + outputs
        if os.path.exists("/checkpoint"):
            mlflow.log_artifacts("/checkpoint", artifact_path="checkpoints")
        if os.path.exists("test_output_images"):
            mlflow.log_artifacts("test_output_images", artifact_path="samples")


if __name__ == "__main__":
    main()
