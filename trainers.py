import torch
from tqdm import tqdm

def train_teacher_mixed_adversarial(model, train_loader, optimizer, criterion, attacker, epochs, device='cuda'):
    model.to(device)
    for epoch in range(epochs):
        model.train() 
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            idx = int(images.size(0) * 0.5)
            clean_images = images[:idx]
            clean_labels = labels[:idx]
            
            adv_images = images[idx:]
            adv_labels = labels[idx:]
            
            adv_images = attacker(adv_images, adv_labels)
            
            mixed_images = torch.cat([clean_images, adv_images], dim=0)
            mixed_labels = torch.cat([clean_labels, adv_labels], dim=0) 
            
            perm = torch.randperm(mixed_images.size(0))
            mixed_images = mixed_images[perm]
            mixed_labels = mixed_labels[perm]
            
            optimizer.zero_grad()    
            outputs = model(mixed_images)   
            loss = criterion(outputs, mixed_labels) 
            loss.backward()             
            optimizer.step()           
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += mixed_labels.size(0)
            correct += (predicted == mixed_labels).sum().item()    
            
            loop.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")


import torch
import torch.optim as optim

def train_student_model(student_combine, criterion, teacher_models, train_loader, device, lr_adv, NUM_EPOCHS=50):
    """
    Hàm huấn luyện mạng Student. 
    Giữ nguyên 100% logic gốc của người dùng.
    """
    optimizer = optim.Adam(
        list(student_combine.parameters()) + list(criterion.parameters()), 
        lr=lr_adv*4
    )
    
    # --- Vòng lặp Epoch (Vòng ngoài) ---
    for epoch in range(NUM_EPOCHS):
        student_combine.train()       
        criterion.train()      
        correct = 0
        total = 0
        running_loss = 0.0     
        running_entropy = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            s_logits, s_features = student_combine(x, return_features=True)
            
            with torch.no_grad():
                t_logits_list = [t(x) for t in teacher_models]
                
            loss, fused_teacher_logits = criterion(s_logits, s_features, t_logits_list, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(s_logits, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
        avg_loss = running_loss / num_batches
        train_acc = correct / total
        epoch_avg_entropy = running_entropy / num_batches
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Acc: {train_acc:.4f} | Loss: {avg_loss:.4f}")
        
    torch.save(student_combine.state_dict(),"./student_combine.pth")
    print("Training Complete!")


# --- Thêm vào file: trainers.py ---

def train_teacher_clean(model, train_loader, optimizer, criterion, epochs, device='cuda'):
    """
    Hàm huấn luyện mô hình cơ sở trên dữ liệu sạch hoàn toàn.
    """
    model.to(device)
    
    for epoch in range(epochs):
        model.train() 
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()         
            outputs = model(images)        
            loss = criterion(outputs, labels) 
            loss.backward()          
            optimizer.step()            
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())
            
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Train Clean Acc: {epoch_acc:.4f}")