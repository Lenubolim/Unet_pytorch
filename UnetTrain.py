import torch
import argparse
from Unet import Unet
from torch import nn
from torch import optim
from UnetDataset import MyData
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([480, 480]),
])
label_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([292, 292]),
])


def train(args):
    model = Unet(3, 1).to(device)
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    mydata = args.dataset
    num_workers = args.num_workers
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    dataset = MyData(mydata, transforms1=image_transform, transforms2=label_transform)
    dataloaders = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,)

    for epoch in range(num_epochs):
        data_size = len(dataloaders.dataset)
        total_loss = 0
        step = 0
        with tqdm(total=(data_size - 1) // batch_size + 1, desc="Epoch {}/{}".format(epoch, num_epochs - 1)) as pbar:
            for img, label in dataloaders:
                step += 1
                inputs = img.to(device)
                labels = label.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        print("Epoch %d loss:%0.2f" % (epoch, total_loss / step))
    torch.save(model.state_dict(), "D:/machine_learning/exercise/UNET/best.pth")
    print("Best model weight has been saved.")
    return model


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--dataset", type=str, default="D:\\machine_learning\\exercise\\UNET\\voc2005_1\\VOC2005_1"
                                                      "\\image")
    parse.add_argument("--batch_size", type=int, default=4)
    parse.add_argument("--num_epochs", type=int, default=200)
    parse.add_argument("--num_workers", type=int, default=1)
    parse.add_argument("--ckpt", type=str, help="model path")
    args = parse.parse_args()
    train(args)
