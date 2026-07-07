import torch.nn as nn
import torch.nn.functional as F
import torch



def KD_1_teacher(s_logits,t_logits, labels, T = 20, alpha = 0.1 ):
    soft_teacher = F.softmax(t_logits / T, dim = 1)
    soft_student = F.log_softmax(s_logits / T, dim = 1)

    distill_loss = nn.KLDivLoss(reduction="batchmean")(soft_student, soft_teacher)
    distill_loss = distill_loss * T * T
    student_loss = F.cross_entropy(s_logits, labels)
    total_loss = alpha * distill_loss + (1 - alpha) * student_loss
    return total_loss
class MLPAttentionDistillationLoss(nn.Module):
    def __init__(self, num_teachers, feature_dim=128, logit_dim=10, hidden_dim=128, T=5, alpha=0.1):
        super(MLPAttentionDistillationLoss, self).__init__()
        self.T = T
        self.alpha = alpha
        
        self.wq = nn.Linear(feature_dim, hidden_dim) # Query Projection
        self.wk = nn.Linear(logit_dim, hidden_dim)   # Key Projection
        self.wv = nn.Linear(logit_dim, hidden_dim)   # Value Projection
        
        self.selector = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_teachers),
            nn.Softmax(dim=1) 
        )
    
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, student_features, teacher_logits_list, labels):
        if isinstance(teacher_logits_list, list):
            teacher_stack = torch.stack(teacher_logits_list, dim=1)
        else:
            teacher_stack = teacher_logits_list
        q = self.wq(student_features).unsqueeze(1) 
        k = self.wk(teacher_stack)
        v = self.wv(teacher_stack)
        
        scores = torch.matmul(q, k.transpose(1, 2)) / (k.size(-1) ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, v).squeeze(1)

        combined = torch.cat([student_features, context], dim=1)
        weights = self.selector(combined) # [Batch, Num_Teachers]

        weighted_teacher_logits = torch.sum(teacher_stack * weights.unsqueeze(-1), dim=1)

        loss_hard = self.ce_loss(student_logits, labels)
        
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(weighted_teacher_logits / self.T, dim=1)
        loss_soft = self.kl_div(soft_student, soft_teacher) * (self.T ** 2)
        
        return self.alpha * loss_hard + (1 - self.alpha) * loss_soft, weighted_teacher_logits


class AblatedMLPDistillationLoss(nn.Module):
    def __init__(self, num_teachers, feature_dim=128, logit_dim=100, 
                 T=20, alpha=0.1):
        super(AblatedMLPDistillationLoss, self).__init__()
        self.num_teachers = num_teachers
        self.logit_dim = logit_dim
        self.T = T
        self.alpha = alpha

        selector_input_dim = feature_dim + (num_teachers * logit_dim)
        
        self.selector_mlp = nn.Sequential(
            nn.Linear(selector_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_teachers),
            nn.Softmax(dim=1)
        )
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, student_features, teacher_logits_list, labels):

        
        if isinstance(teacher_logits_list, list):
            teacher_stack = torch.stack(teacher_logits_list, dim=1)
        else:
            teacher_stack = teacher_logits_list
        batch_size = student_features.size(0)
        teacher_logits_flat = teacher_stack.view(batch_size, -1)
        combined_input = torch.cat([student_features, teacher_logits_flat], dim=1)
        weights = self.selector_mlp(combined_input) # [Batch, Num_Teachers]
        weighted_teacher_logits = torch.sum(teacher_stack * weights.unsqueeze(-1), dim=1)
        loss_hard = self.ce_loss(student_logits, labels)
        
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(weighted_teacher_logits / self.T, dim=1)
        loss_soft = self.kl_div(soft_student, soft_teacher) * (self.T ** 2)
        
        total_loss = self.alpha * loss_hard + (1 - self.alpha) * loss_soft
        return total_loss
    
class AttentionDistillationLoss(nn.Module):
    def __init__(self, num_teachers, feature_dim=128, logit_dim=100, alpha=0.1, T=5):
        super(AttentionDistillationLoss, self).__init__()
        
        self.alpha = alpha
        self.T = T
        d_model = logit_dim 
        
        self.wq = nn.Linear(feature_dim, d_model) 
        self.wk = nn.Linear(logit_dim, d_model)
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, student_features, teacher_logits_list, labels):
        if isinstance(teacher_logits_list, list):
            teacher_stack = torch.stack(teacher_logits_list, dim=1)
        else:
            teacher_stack = teacher_logits_list

        q = self.wq(student_features)
        k = self.wk(teacher_stack)
        
        q_reshaped = q.unsqueeze(1)
        
        scores = torch.matmul(q_reshaped, k.transpose(1, 2))
        
        d_k = k.size(-1)
        scores = scores / (d_k ** 0.5)
        
        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, teacher_stack)
        context = context.squeeze(1) # [Batch, 10]

        loss_student = self.ce_loss(student_logits, labels)
        
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(context / self.T, dim=1)
        
        loss_distill = self.kl_div(soft_student, soft_teacher) * (self.T ** 2)
        
        return self.alpha * loss_student + (1 - self.alpha) * loss_distill, context