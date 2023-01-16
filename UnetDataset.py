import os
from torch.utils.data import Dataset
import PIL.Image as Image


class MyData(Dataset):
    def __init__(self, root, transforms1=None,  transforms2=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __getitem__(self, index):
        image_path = self.imgs[index]
        lable_path = image_path.replace("image", "label")

        image = Image.open(image_path)
        label = Image.open(lable_path)

        if self.transforms1:
            image = self.transforms1(image)
        if self.transforms2:
            label = self.transforms2(label)

        return image, label

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    dataset = MyData("./data/test/image")
