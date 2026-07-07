import torch
import time
from thop import profile

def measure_inference_cost(model, input_size=(1, 1, 28, 28), device='cuda'):
    """
    Hàm đo lường chi phí Suy luận (Inference).
    - Lưu ý: Thay đổi input_size theo dataset của bạn (VD: CIFAR10 là 1,3,32,32)
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_size).to(device)

    # 1. Đo lường số lượng tham số (Parameters)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. Đo lường FLOPs (Floating Point Operations)
    # thop trả về MACs (Multiply-Accumulates). Thường 1 MAC ≈ 2 FLOPs.
    macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2.0

    # 3. Đo lường Độ trễ (Latency)
    # 3.1: Warm-up GPU (Rất quan trọng để đo chính xác)
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)

    # 3.2: Đo thời gian bằng cuda.Event (chuẩn xác nhất cho GPU)
    num_iterations = 100
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]

    with torch.no_grad():
        for i in range(num_iterations):
            start_events[i].record()
            _ = model(dummy_input)
            end_events[i].record()

    torch.cuda.synchronize() # Đợi GPU xử lý xong toàn bộ

    # Tính trung bình thời gian (milliseconds)
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_latency = sum(times) / num_iterations

    return params, flops, avg_latency

def measure_training_overhead(criterion_module):
    """
    Hàm đo lường chi phí tăng thêm lúc Huấn luyện (Training Overhead).
    Chỉ đo các tham số của hàm Loss (Attention, MLP).
    """
    extra_params = sum(p.numel() for p in criterion_module.parameters() if p.requires_grad)
    return extra_params


def test(model_clean, teacher_FGSM = teacher_FGSM,teacher_FFGSM = teacher_FFGSM, teacher_RFGSM =teacher_RFGSM, teacher_PGD = teacher_PGD, test_loader=test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    atk_clean_FGSM = torchattacks.FGSM(teacher_FGSM, eps= 0.1)
    atk_clean_FFGSM = torchattacks.FFGSM(teacher_FFGSM, eps= 0.1)
    atk_clean_RFGSM = torchattacks.RFGSM(teacher_RFGSM, eps= 0.1)
    atk_clean_PGD = torchattacks.PGD(teacher_PGD, eps= 0.1)

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
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model_clean(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

        adv_images = atk_clean_FGSM(images,labels)
        with torch.no_grad():
            outputs = model_clean(adv_images)
            _, predicted = torch.max(outputs, 1)
            val_total_FGSM += labels.size(0)
            val_correct_FGSM += (predicted == labels).sum().item()

        adv_images = atk_clean_FFGSM(images,labels)
        with torch.no_grad():
            outputs = model_clean(adv_images)
            _, predicted = torch.max(outputs, 1)
            val_total_FFGSM += labels.size(0)
            val_correct_FFGSM += (predicted == labels).sum().item()

        adv_images = atk_clean_RFGSM(images,labels)
        with torch.no_grad():
            outputs = model_clean(adv_images)
            _, predicted = torch.max(outputs, 1)
            val_total_RFGSM += labels.size(0)
            val_correct_RFGSM += (predicted == labels).sum().item()

        adv_images = atk_clean_PGD(images,labels)
        with torch.no_grad():
            outputs = model_clean(adv_images)
            _, predicted = torch.max(outputs, 1)
            val_total_PGD += labels.size(0)
            val_correct_PGD += (predicted == labels).sum().item()


    print(f"Epoch Test CLEAN: {val_correct/val_total:.4f}")
    print(f"Epoch Test FGSM: {val_correct_FGSM/val_total_FGSM:.4f}")
    print(f"Epoch Test FFGSM: {val_correct_FFGSM/val_total_FFGSM:.4f}")
    print(f"Epoch Test RFGSM: {val_correct_RFGSM/val_total_RFGSM:.4f}")
    print(f"Epoch Test PGD: {val_correct_PGD/val_total_PGD:.4f}")

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionDistillationLoss(nn.Module):
    def __init__(self, num_teachers, feature_dim=128, logit_dim=100, alpha=0.1, T=T):
        super(AttentionDistillationLoss, self).__init__()
        
        self.alpha = alpha
        self.T = T
        d_model = logit_dim 
        
        # --- ATTENTION PROJECTIONS ---
        self.wq = nn.Linear(feature_dim, d_model) 
        self.wk = nn.Linear(logit_dim, d_model)
        
        # [ĐÃ XÓA LỚP wv] Dùng trực tiếp Logits của Teacher để tránh làm hỏng phân phối
        
        # --- LOSS FUNCTIONS ---
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, student_features, teacher_logits_list, labels):
        # 1. Stack Teacher Logits -> [Batch, Num_Teachers, Classes]
        if isinstance(teacher_logits_list, list):
            teacher_stack = torch.stack(teacher_logits_list, dim=1)
        else:
            teacher_stack = teacher_logits_list
            
        # ==========================================
        # LOGIC ATTENTION (Feature-Based Query)
        # ==========================================
        q = self.wq(student_features)
        k = self.wk(teacher_stack)
        
        # Expand Q: [Batch, 10] -> [Batch, 1, 10]
        q_reshaped = q.unsqueeze(1)
        
        # Matmul: [Batch, 1, 10] x [Batch, 10, N] -> [Batch, 1, N]
        scores = torch.matmul(q_reshaped, k.transpose(1, 2))
        
        # Scale
        d_k = k.size(-1)
        scores = scores / (d_k ** 0.5)
        
        # Softmax -> Weights [Batch, 1, N]
        attn_weights = F.softmax(scores, dim=-1)
        
        # [ĐÃ SỬA] Nhân trực tiếp Attention Weights với Logits gốc của Teacher
        # [Batch, 1, N] x [Batch, N, 10] -> [Batch, 1, 10]
        context = torch.matmul(attn_weights, teacher_stack)
        context = context.squeeze(1) # [Batch, 10]
        
        # ==========================================
        # TÍNH LOSS
        # ==========================================
        loss_student = self.ce_loss(student_logits, labels)
        
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(context / self.T, dim=1)
        
        loss_distill = self.kl_div(soft_student, soft_teacher) * (self.T ** 2)
        
        return self.alpha * loss_student + (1 - self.alpha) * loss_distill, context
    



class MLPAttentionDistillationLoss(nn.Module):
    def __init__(self, num_teachers, feature_dim=128, logit_dim=100, hidden_dim=128, T=T, alpha=0.1):
        super(MLPAttentionDistillationLoss, self).__init__()
        self.T = T
        self.alpha = alpha
        
        # === PHẦN 1: ATTENTION LAYERS (Tương đương class AttentionLayer cũ) ===
        self.wq = nn.Linear(feature_dim, hidden_dim) # Query Projection
        self.wk = nn.Linear(logit_dim, hidden_dim)   # Key Projection
        self.wv = nn.Linear(logit_dim, hidden_dim)   # Value Projection
        
        # === PHẦN 2: SELECTOR MLP (Tương đương self.teacher_selector cũ) ===
        # Input: Feature gốc + Feature ngữ cảnh từ Attention
        self.selector = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_teachers),
            nn.Softmax(dim=1) # Ra xác suất chọn teacher
        )
        
        # Các hàm loss cơ bản
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, student_features, teacher_logits_list, labels):
        # 1. Chuẩn bị Teacher Logits: [Batch, Num_Teachers, Classes]
        # (Tương đương: tf.stack(teacher_logits_list, axis=1))
        if isinstance(teacher_logits_list, list):
            teacher_stack = torch.stack(teacher_logits_list, dim=1)
        else:
            teacher_stack = teacher_logits_list

        # 2. Tính Attention (Tương đương hàm call() của AttentionLayer)
        # Query [B, 1, H]
        q = self.wq(student_features).unsqueeze(1) 
        # Key [B, N, H]
        k = self.wk(teacher_stack)
        v = self.wv(teacher_stack)
        
        # Score = Q * K^T (Tương đương tf.matmul(..., transpose_b=True))
        scores = torch.matmul(q, k.transpose(1, 2)) / (k.size(-1) ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Context = Weights * V (Tương đương tf.matmul(attention_weights, v))
        context = torch.matmul(attn_weights, v).squeeze(1)

        # 3. Tính Selector (Tương đương teacher_selector(combined_features))
        combined = torch.cat([student_features, context], dim=1)
        weights = self.selector(combined) # [Batch, Num_Teachers]

        # 4. Tổng hợp Teacher (Tương đương tf.reduce_sum(...))
        # Nhân broadcasting: [B, N, C] * [B, N, 1]
        weighted_teacher_logits = torch.sum(teacher_stack * weights.unsqueeze(-1), dim=1)

        # 5. Tính Loss (Tương đương phần cuối train_step)
        loss_hard = self.ce_loss(student_logits, labels)
        
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(weighted_teacher_logits / self.T, dim=1)
        loss_soft = self.kl_div(soft_student, soft_teacher) * (self.T ** 2)
        
        return self.alpha * loss_hard + (1 - self.alpha) * loss_soft, weighted_teacher_logits


import torch
import torch.nn.functional as F

def calculate_average_entropy(fused_targets, is_logits=True, temperature=T):
    """
    Tính toán Entropy trung bình của phân phối mục tiêu tổng hợp từ các Teacher.
    
    Args:
        fused_targets (torch.Tensor): Tensor chứa mục tiêu tổng hợp (kích thước: [batch_size, num_classes]).
        is_logits (bool): True nếu đầu vào là Logits, False nếu đầu vào đã là Xác suất (Probabilities).
        temperature (float): Nhiệt độ T dùng để làm mềm (soften) phân phối. (Mặc định = 1.0).
        
    Returns:
        float: Giá trị Entropy trung bình của cả batch.
    """
    with torch.no_grad(): # Không cần tính gradient cho việc đo lường metric
        if is_logits:
            # Nếu là Logits: Áp dụng Temperature và Softmax
            # Dùng log_softmax kết hợp softmax để tính toán ổn định, tránh lỗi NaN
            soft_probs = F.softmax(fused_targets / temperature, dim=1)
            log_soft_probs = F.log_softmax(fused_targets / temperature, dim=1)
        else:
            # Nếu đã là Xác suất (Soft targets)
            soft_probs = fused_targets
            # Cộng thêm một giá trị epsilon cực nhỏ (1e-8) để tránh lỗi log(0)
            log_soft_probs = torch.log(soft_probs + 1e-8)
            
        # Tính entropy cho từng mẫu trong batch: H(P) = -sum(P * log(P))
        # log ở đây là logarit tự nhiên (cơ số e). Đơn vị là 'nats'.
        entropy = -(soft_probs * log_soft_probs).sum(dim=1)
        
        # Lấy giá trị trung bình trên toàn bộ batch
        avg_entropy = entropy.mean().item()
        
    return avg_entropy
import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    """
    Thiết lập seed cho toàn bộ hệ thống để đảm bảo tính tái tạo (Reproducibility).
    """
    # 1. Cố định seed cho Python built-in random
    random.seed(seed)
    
    # 2. Cố định seed cho thư viện toán học NumPy
    np.random.seed(seed)
    
    # 3. Cố định seed cho PyTorch (CPU)
    torch.manual_seed(seed)
    
    # 4. Cố định seed cho PyTorch (GPU/CUDA)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Bắt buộc nếu bạn dùng nhiều GPU (Multi-GPU)
    
    # 5. Ép CUDNN (Backend của GPU) chạy ở chế độ tất định (Deterministic)
    # Lưu ý: Việc này có thể làm giảm tốc độ huấn luyện đi một chút xíu, 
    # nhưng là BẮT BUỘC để kết quả sinh nhiễu và huấn luyện không bị chệch.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 6. Cố định Hash Seed của môi trường hệ điều hành
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Đã khóa cứng seed = {seed} cho toàn bộ môi trường huấn luyện!")

# --- GỌI HÀM NGAY LÚC KHỞI TẠO ---
set_seed(42)