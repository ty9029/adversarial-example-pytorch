import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_dataset(data_name, data_root, image_size, train):
    if data_name == "lsun":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()])

    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()])

    if data_name == "mnist":
        dataset = datasets.MNIST(root=data_root, train=train, transform=transform, download=True)

    elif data_name == "fushion-mnist":
        dataset = datasets.FashionMNIST(root=data_root, train=train, transform=transform, download=True)

    elif data_name == "kmnist":
        dataset = datasets.KMNIST(root=data_root, train=train, transform=transform, download=True)

    elif data_name == "emnist":
        dataset = datasets.EMNIST(root=data_root, split="balanced", train=train, transform=transform, download=True)

    elif data_name == "cifar10":
        dataset = datasets.CIFAR10(root=data_root, train=train, transform=transform, download=True)

    elif data_name == "cifar100":
        dataset = datasets.CIFAR100(root=data_root, train=train, transform=transform, download=True)

    elif data_name == "lsun":
        dataset = datasets.LSUN(root=data_root, classes="train", transform=transforms)

    elif data_name == "stl10":
        dataset = datasets.STL10(root=data_root, split="train", transform=transform, download=True)

    else:
        dataset = None

    return dataset
