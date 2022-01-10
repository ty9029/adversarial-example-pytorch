import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import CNN
from dataset import get_dataset
from utils import plot_result


def targeted_fast_gradient(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image - epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def test(model, opt):
    criterion = nn.CrossEntropyLoss()

    test_dataset = get_dataset(opt.data_name, opt.data_root, opt.image_size, train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    for i, (images, cur_labels) in enumerate(test_loader):
        labels = torch.LongTensor([int(input("Current Label: {}, Target Label : ".format(cur_labels[0])))])
        labels = labels.to(opt.device)
        images = images.to(opt.device)
        images.requires_grad = True

        model.zero_grad()
        outputs = model(images)
        result = F.softmax(outputs, dim=1)
        loss = criterion(outputs, labels)
        loss.backward()

        images_grad = images.grad.data
        perturbed_images = targeted_fast_gradient(images, opt.epsilon, images_grad)
        perturbed_outputs = model(perturbed_images)
        attack_result = F.softmax(perturbed_outputs, dim=1)

        images = images.permute(0, 2, 3, 1).detach().cpu().numpy()
        result = result[0].detach().cpu().numpy()

        perturbed_images = perturbed_images.permute(0, 2, 3, 1).detach().cpu().numpy()
        attack_result = attack_result[0].cpu().detach().numpy()

        plot_result("./outputs/targeted_fgsm/{}/{}.png".format(opt.data_name, i), images, result, perturbed_images, attack_result)


def main():
    parser = argparse.ArgumentParser(description="T-FGSM")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--data_name", type=str, default="mnist")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--image_channels", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--epsilon", type=int, default=0.3)
    opt = parser.parse_args()

    os.makedirs("./outputs/targeted_fgsm/{}".format(opt.data_name))

    model = CNN(opt.image_size, opt.image_channels).to(opt.device)
    model.load_state_dict(torch.load("./weights/{}.pth".format(opt.data_name)))

    test(model, opt)


if __name__ == '__main__':
    main()
