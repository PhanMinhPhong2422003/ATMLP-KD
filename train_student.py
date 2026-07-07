import torch
import os
from config import *
from dataset import get_cifar10_loaders
from model import SimpleCNN
from losses import MLPAttentionDistillationLoss
from eval import test
from trainers import train_student_model  # Nạp hàm huấn luyện từ trainers.py
def main():
    print("Đang chuẩn bị DataLoader...")
    train_loader, test_loader = get_cifar10_loaders()
    
    print("Đang nạp trọng số cho các Teacher Models...")
    teacher_names = ['FGSM', 'FFGSM', 'RFGSM', 'PGD']
    teacher_models = []
    
    for name in teacher_names:
        t_model = SimpleCNN(width=width, height=height, depth=depth, classes=classes).to(device)
        weight_path = f"./teacher_{name}.pth"
        
        if os.path.exists(weight_path):
            t_model.load_state_dict(torch.load(weight_path, map_location=device))
        else:
            print(f"Cảnh báo: Không tìm thấy file {weight_path}!")
            
        t_model.eval() 
        teacher_models.append(t_model)
        
    t_fgsm, t_ffgsm, t_rfgsm, t_pgd = teacher_models

    student_combine = SimpleCNN(width=width, height=height, depth=depth, classes=classes).to(device)
    criterion = MLPAttentionDistillationLoss(num_teachers=4, T=5, alpha=0.1).to(device)

    train_student_model(
        student_combine=student_combine,
        criterion=criterion,
        teacher_models=teacher_models,
        train_loader=train_loader,
        device=device,
        lr_adv=lr_adv,
        NUM_EPOCHS=EPOCHS_STUDENT
    )
    
if __name__ == "__main__":
    main()