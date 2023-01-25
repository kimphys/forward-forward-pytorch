import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision

import numpy as np
import pandas as pd

from dataset import MNISTDataset, embedding
from model import FFNetwork


if __name__ == "__main__":

    num_classes = 10
    batch_size = 10000

    df_train = pd.read_csv('./sample_data/mnist_train_small.csv', header=None)
    df_test = pd.read_csv('./sample_data/mnist_test.csv', header=None)

    train_labels = df_train.iloc[:, 0]
    train_images = df_train.iloc[:, 1:]
    test_labels = df_test.iloc[:, 0]
    test_images = df_test.iloc[:, 1:]

    custom_transform = torchvision.transforms.Compose(
                        [
                        torchvision.transforms.ToPILImage(),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5, ), (0.5, ))
                    ])

    train_data = MNISTDataset(train_images, train_labels, transforms=custom_transform)
    test_data = MNISTDataset(test_images, test_labels, transforms=custom_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    model = FFNetwork().cuda() if torch.cuda.is_available() else FFNetwork()

    print("Start training...")
    for data, lbl in train_loader:

        data_pos = data.clone().detach()
        lbl_pos = lbl.clone().detach()
        data_pos = embedding(data_pos, lbl, num_classes=num_classes)
        
        data_neg = data.clone().detach()
        lbl_neg = torch.from_numpy(np.random.choice(num_classes, data.shape[0]))
        data_neg = embedding(data_neg, lbl_neg, num_classes=num_classes)

        if torch.cuda.is_available():
            data_pos, data_neg = data_pos.cuda(), data_neg.cuda()

        model.train(data_pos, data_neg)

    print("Start testing...")
    predictions = []
    groundtruths = []

    for i, (data_test, lbl_test) in enumerate(test_loader):
        if torch.cuda.is_available():
            data_test = data_test.cuda()

        prediction = model.predict(data_test).item()
        groundtruth = lbl_test.item()

        predictions.append(prediction)
        groundtruths.append(groundtruth)

    from sklearn.metrics import f1_score
    print("F1-score: ", f1_score(groundtruths, predictions, average='macro'))