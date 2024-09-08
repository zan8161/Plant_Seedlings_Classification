import sys
import torch
import torch.nn as nn
from torchvision import models

import pandas as pd
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import random_split

import torchvision.transforms.transforms as T

from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# This function is used to gen a .csv file contains labels of the images
# The format of the file is like:

# Labels        index
# Black-grass   0
# ...           ...
# Sugar beet    11

def gen_image_labels(img_dir):
    if os.path.exists(os.path.join(img_dir, "labels.csv")):
        return
    dirs = os.listdir(img_dir)
    labels = []
    labels_idx = []
    for dir in dirs:
        for folder, subfolders, files in os.walk(os.path.join(img_dir, dir)):
            for file in files:
                labels.append(dir)
                labels_idx.append(dirs.index(dir))

    label_sheet = [labels, labels_idx]
    df = pd.DataFrame(label_sheet)
    df = df.T
    df.to_csv(os.path.join(img_dir, "labels.csv"),
              index=False, header=False)


def draw_loss_curve(train_loss, valid_loss):
    save_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "result")
    if os.path.exists(save_dir) != True:
        os.makedirs(save_dir)

    plt.plot(train_loss, 'b', label="train loss")
    plt.plot(valid_loss, 'r', label="valid loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss.png"))


def draw_confusion_matrix(cm):
    cm_display = ConfusionMatrixDisplay(cm).plot(
        cmap=plt.cm.Blues, values_format='g')
    save_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "result")
    if os.path.exists(save_dir) != True:
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))


class TrainDataSet(Dataset):
    def __init__(self, root, transform=None, outdim=12):
        self.root = root
        self.img_dir = os.path.join(self.root, "train")

        self.images = [file for folder, subfolders, files in os.walk(
            self.img_dir) for file in files]

        self.labels = pd.read_csv(os.path.join(
            self.root, "labels.csv"), header=None)

        self.transform = transform
        self.outdim = outdim

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir, self.labels.iloc[idx, 0], self.images[idx])
        image = Image.open(img_path).convert('RGB')
        img_label_idx = self.labels.iloc[idx, 1]

        image_label = torch.zeros(self.outdim)
        image_label[img_label_idx] = 1

        if self.transform:
            image = self.transform(image)

        return image, image_label


class TestDataSet(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.img_dir = os.path.join(self.root, "test")
        self.images = [file for folder, subfolders,
                       files in os.walk(self.img_dir)for file in files]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


def train(model, train_loader, valid_loader, device, n_epoch):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    train_loss = []
    valid_loss = []
    best_loss = sys.maxsize
    best_matrix = None

    for epoch in range(n_epoch):
        """ Training """
        model.train()
        train_loss_record = []
        """ Visualize the training progress """
        train_bar = tqdm(train_loader, position=0, leave=True)
        train_bar.set_description(f"Epoch [{epoch+1} / {n_epoch}] Training")

        for data, target in train_bar:
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            pred = model(data)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            train_loss_record.append(loss.detach().item())
            train_bar.set_postfix({"loss": loss.detach().item()})

        mean_train_loss = sum(train_loss_record) / len(train_loss_record)
        train_loss.append(mean_train_loss)
        print(f"Mean train loss in Epoch {epoch+1} : {mean_train_loss:.4f}")

        """ Validation """
        model.eval()
        valid_loss_record = []
        # These two lists are used to provide info of a confusion matrix
        valid_labels = []
        valid_preds = []

        """ Visualize the validating progress """
        valid_bar = tqdm(valid_loader, position=0, leave=True)
        valid_bar.set_description(f"Epoch [{epoch+1} / {n_epoch}] Validating")

        for data, target in valid_bar:
            data = data.to(device)
            target = target.to(device)
            with torch.no_grad():
                pred = model(data)
                loss = criterion(pred, target)
            valid_bar.set_postfix({"loss": loss.detach().item()})
            valid_loss_record.append(loss.detach().item())
            # convert the type of pred result to list
            _, preds = torch.max(pred.data, 1)
            _, targets = torch.max(target, 1)
            valid_preds.extend(preds.cpu().numpy())
            valid_labels.extend(targets.cpu().numpy())

        mean_valid_loss = sum(valid_loss_record) / len(valid_loss_record)
        valid_loss.append(mean_valid_loss)
        print(f"Mean valid loss in Epoch {epoch+1} : {mean_valid_loss:.4f}")

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            # save the best model with a smallest valid loss
            print("Saving the best model...")
            save_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "model")
            if os.path.exists(save_path) != True:
                os.makedirs(save_path)
            torch.save(model.state_dict(),
                       os.path.join(save_path, "model.ckpt"))
            # save the confusion matrix with a smallest valid loss
            print("Saving the best confusion matrix...")
            best_matrix = confusion_matrix(valid_labels, valid_preds)

    draw_loss_curve(train_loss, valid_loss)
    draw_confusion_matrix(best_matrix)


def test(model, test_loader, device):
    # load the best model saved in the training process
    model_save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model")
    # load the model in the "model" dir
    model.load_state_dict(torch.load(
        os.path.join(model_save_path, "model.ckpt")))
    model = model.to(device)
    model.eval()

    classes = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common wheat", "Fat Hen",
               "Loose Silky-bent", "Maize", "Scentless Mayweed", "Shepherds Purse", "Small-flowered Cranesbill", "Sugar beet"]

    pred_result = []

    """ Visualize the testing progress """
    test_bar = tqdm(test_loader, position=0, leave=True)

    for data in test_bar:
        data = data.to(device)
        out = model(data)
        # cuz I set the batch size of the testloader to 1,
        # I can just obtained test results by argmax() function
        pred = out.argmax(dim=1)

        test_bar.set_description("Testing...")

        pred_result.append(classes[pred])

    return pred_result


def make_submission_sample(test_img_dir, pred_result):
    images = [file for folder, subfolders,
              files in os.walk(test_img_dir) for file in files]

    submission_sample = {'file': images, 'species': pred_result}

    df = pd.DataFrame(submission_sample)
    save_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "result")
    if os.path.exists(save_path) != True:
        os.makedirs(save_path)
    df.to_csv(os.path.join(save_path, "pred.csv"), index=False)


""" Main """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 25

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

model = models.resnet50()
# modify the fc layer of the resnet50
in_features = model.fc.in_features
num_class = 12
model.fc = nn.Linear(in_features, num_class)

gen_image_labels(root)
trainset = TrainDataSet(root=root, transform=transforms)
testset = TestDataSet(root=root, transform=transforms)

validset_size = int(len(trainset) * 0.2)
trainset_size = len(trainset) - validset_size

trainset, validset = random_split(
    trainset,
    [trainset_size, validset_size],
    generator=torch.Generator().manual_seed(8161))

TrainLoader = DataLoader(trainset, batch_size=8, shuffle=True)
ValidLoader = DataLoader(validset, batch_size=8, shuffle=False)
TestLoader = DataLoader(testset, batch_size=1, shuffle=False)

train(model, TrainLoader, ValidLoader, device, epochs)
pred_result = test(model, TestLoader, device)

test_img_dir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "Data", "test")
make_submission_sample(test_img_dir, pred_result)
