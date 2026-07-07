# --- File: train_teachers.py ---
import torch
import torch.nn as nn
import torch.optim as optim
import torchattacks

from config import *
from dataset import get_cifar10_loaders
from model import SimpleCNN
from trainers import train_teacher_mixed_adversarial, train_teacher_clean # <--- Import hàm từ trainers.py

def main():

    set_seed(42)
    train_loader, test_loader = get_cifar10_loaders()
    model_clean = SimpleCNN(width=width, height=height, depth=depth, classes=classes).to(device)
    optimizer_clean = optim.Adam(model_clean.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Huấn luyện 10 epochs theo ý tưởng của bạn
    train_teacher_clean(
        model=model_clean,
        train_loader=train_loader,
        optimizer=optimizer_clean,
        criterion=criterion,
        epochs=3,
        device=device
    )
    torch.save(model_clean.state_dict(), "./clean.pth")
    
    teacher_names = ["FGSM", "FFGSM", "RFGSM", "PGD"]

    for name in teacher_names:
        print(f"\n{'='*50}")
        print(f"🚀 BẮT ĐẦU HUẤN LUYỆN TEACHER: {name}")
        print(f"{'='*50}")
        set_seed(42)

        model = SimpleCNN(width=width, height=height, depth=depth, classes=classes).to(device)
        model.load_state_dict(torch.load("./clean.pth")) 
        
        optimizer_adv = optim.Adam(model.parameters(), lr=5e-4)
        
        if name == "FGSM":
            attacker = torchattacks.FGSM(model, eps=0.1)
        elif name == "FFGSM":
            attacker = torchattacks.FFGSM(model, eps=0.1)
        elif name == "RFGSM":
            attacker = torchattacks.RFGSM(model, eps=0.1)
        elif name == "PGD":
            attacker = torchattacks.PGD(model, eps=0.1, steps=10)
        
        train_teacher_mixed_adversarial(
            model=model, 
            train_loader=train_loader, 
            optimizer=optimizer_adv, 
            criterion=criterion, 
            attacker=attacker, 
            epochs=3, 
            device=device
        )
        
        save_path = f"./teacher_{name}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"✅ Đã lưu thành công: {save_path}")

if __name__ == "__main__":
    main()