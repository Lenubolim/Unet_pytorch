import torch
import argparse
from Unet import Unet
from UnetDataset import MyData
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([480, 480]),
])
label_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([292, 292]),
])


def test(args):
    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.weight, map_location='cpu'))
    dataset = MyData(args.dataset, transforms1=image_transform, transforms2=label_transform)
    dataloaders = DataLoader(dataset, batch_size=1)
    model.eval()
    with torch.no_grad():
        for img, _ in dataloaders:
            predict = model(img)
            label = torch.squeeze(predict).numpy()
            plt.imshow(label)
            plt.pause(1)
        plt.show()


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--dataset", type=str, default="D:\\machine_learning\\exercise\\UNET\\voc2005_1\\VOC2005_1"
                                                      "\\image")
    parse.add_argument("--weight", type=str, default="D:/machine_learning/exercise/UNET/9.pth", help="model weight path")
    args = parse.parse_args()
    test(args)
