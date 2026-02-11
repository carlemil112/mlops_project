import torch
from torch import nn
from data_loader import *
from torch import optim
from collections import OrderedDict
from torch.optim.lr_scheduler import StepLR
from models.discriminator import *


def load_random(path, model):
    """loading a checkpoint"""
    checkpoint = torch.load(path, weights_only=False, map_location=torch.device("cpu"))
    state_dict = checkpoint["model_state_dict"]

    # FROM CHATGBT
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # strip only at the start
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)

    epoch = checkpoint.get("epoch", 0)
    val_loss = checkpoint.get("val_loss", None)

    return model, epoch, val_loss


def load_pretrained(path, model):
    """loading a checkpoint"""
    checkpoint = torch.load(path, weights_only=False, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    val_loss = checkpoint["val_loss"]

    return model, epoch, val_loss


def load_patch(path, model):
    """loading a checkpoint"""
    checkpoint = torch.load(path, weights_only=False, map_location=torch.device("cpu"))

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

    epoch = checkpoint["epoch"]
    val_loss = checkpoint["val_loss"]

    return model, epoch, val_loss


def test(model, device, val_loader):
    """validation loop"""
    model.eval()
    lossfunction = nn.L1Loss()
    total_loss = 0
    total_batch_size = 0
    # psnr_metric = PSNR(data_range=1.0)

    with torch.no_grad():
        loss = 0
        for batch, (x, y) in enumerate(val_loader):
            if batch % 5000 == 0:
                print(f"Progress: {batch}/328500")
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            batch_loss = lossfunction(y, output).item()
            batch_size = x.size(0)

            total_loss += batch_loss * batch_size
            total_batch_size += batch_size

        loss = total_loss / total_batch_size
        print("MAE: ", loss, "\n")
    return loss


def save_test_image(pred_tensor, target_tensor, i, model):
    pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)
    pred_img = transforms.functional.to_pil_image(pred_tensor[0].detach().cpu())
    pred_img.save(f"test_output_images/{model}_pred_img{i}.jpg")
    target_tensor = torch.clamp(target_tensor, 0.0, 1.0)
    target_img = transforms.functional.to_pil_image(target_tensor[0].detach().cpu())
    target_img.save(f"test_output_images/{model}_y_img{i}.jpg")


def main():
    # import model
    torch.manual_seed(42)
    model_name = "patch_unet"  # random_unet, pretrained_unet, patch_unet
    data_path = "data"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
        print("mps")
    else:
        device = torch.device("cpu")
        print("cpu")

    if model_name == "random_unet":
        from models.unet import Unet as unet

        model = unet()
        checkpoint_path = "checkpoint/random_unet.tar"
        model = model.to(device)
        model, epoch, val_loss = load_random(checkpoint_path, model)
    elif model_name == "pretrained_unet":
        model = torch.hub.load(
            "milesial/Pytorch-UNet", "unet_carvana", pretrained=False, scale=1
        )
        model.inc.double_conv[0] = nn.Conv2d(
            1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        model.outc.conv = nn.Conv2d(
            64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        checkpoint_path = "checkpoint/pretrained_unet.tar"
        model = model.to(device)
        model, epoch, val_loss = load_pretrained(checkpoint_path, model)
    elif model_name == "patch_unet":
        from models.unet import Unet as unet

        model = unet()
        checkpoint_path = "checkpoint/patch_unet.tar"
        model = model.to(device)
        model, epoch, val_loss = load_patch(checkpoint_path, model)
    else:
        print("Choose one of 3 models: random_unet, pretrained_unet, patch_unet")

    print("checkpoint loaded")
    print(f"Checkpoint after {epoch}, with validation loss {val_loss}")

    test_data = gray_color_data(data_path, split="test", train=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=0
    )
    print("data loaded")
    # loss = test(model, device, test_loader)
    # print(loss)

    for i, (x, y) in enumerate(test_loader):
        test_img = x.to(device)
        model.eval()
        test_pred = model(test_img)
        save_test_image(test_pred, y, i, model_name)
        if i > 10:
            break


if __name__ == "__main__":
    main()
