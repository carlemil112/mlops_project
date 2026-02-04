from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class gray_color_data(Dataset):
    def __init__(self, path, split, train):
        # dataset
        self.data = datasets.Places365(path, split=split, small = True,  download=True, transform = None)

        # trainset or not?
        self.train = train

        # general augmentation
        self.data_aug = transforms.Compose([
            transforms.RandomCrop((128,128)),
            transforms.RandomHorizontalFlip(p = 0.25),
            transforms.RandomVerticalFlip(p = 0.25),
            transforms.ColorJitter(brightness = 0.5),
            ])

        # input = GRAY
        self.transforminput = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        # target = RGB
        self.transformtarget = transforms.Compose([
            transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, _ = self.data[idx]
        if self.train == True:
            img = self.data_aug(img)

        x = self.transforminput(img)   # grayscale tensor
        y = self.transformtarget(img)  # RGB tensor
        return x, y
