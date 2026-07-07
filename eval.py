import torch
import torchattacks

def test(model_clean, teacher_FGSM, teacher_FFGSM, teacher_RFGSM, teacher_PGD, test_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Khởi tạo các bộ tấn công
    atk_clean_FGSM = torchattacks.FGSM(teacher_FGSM, eps=0.1)
    atk_clean_FFGSM = torchattacks.FFGSM(teacher_FFGSM, eps=0.1)
    atk_clean_RFGSM = torchattacks.RFGSM(teacher_RFGSM, eps=0.1)
    atk_clean_PGD = torchattacks.PGD(teacher_PGD, eps=0.1)
    
    model_clean.eval()
    
    val_correct = 0
    val_total = 0
    
    val_correct_FGSM = 0
    val_total_FGSM = 0
    
    val_correct_FFGSM = 0
    val_total_FFGSM = 0
    
    val_correct_RFGSM = 0
    val_total_RFGSM = 0
    
    val_correct_PGD = 0
    val_total_PGD = 0
    
    print("\nĐang chạy kiểm thử (Test)...")
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 1. Đánh giá CLEAN
        with torch.no_grad():
            outputs = model_clean(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

        # 2. Đánh giá FGSM
        adv_images = atk_clean_FGSM(images, labels)
        with torch.no_grad():
            outputs = model_clean(adv_images)
            _, predicted = torch.max(outputs, 1)
            val_total_FGSM += labels.size(0)
            val_correct_FGSM += (predicted == labels).sum().item()

        # 3. Đánh giá FFGSM
        adv_images = atk_clean_FFGSM(images, labels)
        with torch.no_grad():
            outputs = model_clean(adv_images)
            _, predicted = torch.max(outputs, 1)
            val_total_FFGSM += labels.size(0)
            val_correct_FFGSM += (predicted == labels).sum().item()

        # 4. Đánh giá RFGSM
        adv_images = atk_clean_RFGSM(images, labels)
        with torch.no_grad():
            outputs = model_clean(adv_images)
            _, predicted = torch.max(outputs, 1)
            val_total_RFGSM += labels.size(0)
            val_correct_RFGSM += (predicted == labels).sum().item()

        # 5. Đánh giá PGD
        adv_images = atk_clean_PGD(images, labels)
        with torch.no_grad():
            outputs = model_clean(adv_images)
            _, predicted = torch.max(outputs, 1)
            val_total_PGD += labels.size(0)
            val_correct_PGD += (predicted == labels).sum().item()

    # In kết quả
    print("="*40)
    print(f"Epoch Test CLEAN: {val_correct/val_total:.4f}")
    print(f"Epoch Test FGSM: {val_correct_FGSM/val_total_FGSM:.4f}")
    print(f"Epoch Test FFGSM: {val_correct_FFGSM/val_total_FFGSM:.4f}")
    print(f"Epoch Test RFGSM: {val_correct_RFGSM/val_total_RFGSM:.4f}")
    print(f"Epoch Test PGD: {val_correct_PGD/val_total_PGD:.4f}")
    print("="*40)