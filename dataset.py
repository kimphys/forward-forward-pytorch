from torch.nn import functional as F
from torch.utils.data import Dataset
import torchvision

import numpy as np



def embedding(data, y, num_classes):
    input = data.clone().detach()
    one_hot = F.one_hot(y, num_classes=num_classes)
    input[:,:,0,:num_classes] = one_hot.view(one_hot.shape[0], 1, one_hot.shape[1])
    return input

class MNISTDataset(Dataset):
    def __init__(self, images, labels, size=(28, 28), num_classes=10, transforms=None, train=True):
        self.X = images
        self.y = labels

        self.w = size[0]
        self.h = size[1]

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = torchvision.transforms.Compose(
                                  [
                                  torchvision.transforms.ToPILImage(),
                                  torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize((0.5, ), (0.5, ))
                              ])
            
        self.train = train
         
    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X.iloc[i, :]
        data = np.asarray(data).astype(np.uint8).reshape(self.h, self.w, 1)
        
        if self.transforms:
            data = self.transforms(data)

        return data, self.y[i]