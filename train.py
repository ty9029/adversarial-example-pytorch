import argparse
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import get_dataset
from models import CNN


def train(model, optimizer, train_loader, opt):
    criterion = nn.CrossEntropyLoss()

    model.train()
    for image, target in train_loader:
        image, target = image.to(opt.device), target.to(opt.device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    return loss


def main():
    parser = argparse.ArgumentParser(description="CNN")
    parser.add_argument("--num_epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--data_name", type=str, default="mnist")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--image_channels", type=int, default=1)
    opt = parser.parse_args()

    os.makedirs("./weights", exist_ok=True)

    model = CNN(opt.image_size, opt.image_channels).to(opt.device)
    optimizer = Adam(model.parameters(), lr=0.001)

    train_dataset = get_dataset(opt.data_name, opt.data_root, opt.image_size, train=True)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

    for epoch in range(opt.num_epoch):
        loss = train(model, optimizer, train_loader, opt)
        print("epoch: {:04}, loss: {:.6f}".format(epoch, loss))

    torch.save(model.state_dict(), "./weights/{}.pth".format(opt.data_name))


if __name__ == "__main__":
    main()
