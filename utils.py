import torch
import time
from thop import profile
import torch
import torch.nn.functional as F
def measure_inference_cost(model, input_size=(1, 1, 28, 28), device='cuda'):

    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_size).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2.0

    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
    num_iterations = 100
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iterations)]
    with torch.no_grad():
        for i in range(num_iterations):
            start_events[i].record()
            _ = model(dummy_input)
            end_events[i].record()
    torch.cuda.synchronize() 
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_latency = sum(times) / num_iterations
    return params, flops, avg_latency

def measure_training_overhead(criterion_module):
    extra_params = sum(p.numel() for p in criterion_module.parameters() if p.requires_grad)
    return extra_params


def calculate_average_entropy(fused_targets, is_logits=True, temperature = 5):
    with torch.no_grad(): 
        if is_logits:

            soft_probs = F.softmax(fused_targets / temperature, dim=1)
            log_soft_probs = F.log_softmax(fused_targets / temperature, dim=1)
        else:
            soft_probs = fused_targets
            log_soft_probs = torch.log(soft_probs + 1e-8)
        entropy = -(soft_probs * log_soft_probs).sum(dim=1)        
        avg_entropy = entropy.mean().item()
        
    return avg_entropy