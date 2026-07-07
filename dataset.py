import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import BATCH_SIZE

def get_cifar10_loaders():
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Khởi tạo Generator để đảm bảo quá trình xáo trộn (shuffle) giống nhau qua các lần chạy
    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id),
        generator=g,
        num_workers=8
    )
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    
    return train_loader, test_loader


def get_mnist_loaders():
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Khởi tạo Generator để đảm bảo quá trình xáo trộn (shuffle) giống nhau qua các lần chạy
    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id),
        generator=g,
        num_workers=8
    )
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    
    return train_loader, test_loader



def get_fmnist_loaders():
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Khởi tạo Generator để đảm bảo quá trình xáo trộn (shuffle) giống nhau qua các lần chạy
    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id),
        generator=g,
        num_workers=8
    )
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    
    return train_loader, test_loader