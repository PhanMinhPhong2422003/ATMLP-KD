import torch
import random
import numpy as np
import os

BATCH_SIZE = 64
LR_TEACHER = 5e-4
LR_STUDENT = 1e-3
EPOCHS_STUDENT = 50
TEMPERATURE = 5
ALPHA = 0.1  
width = 32
height = 32
depth = 3
classes = 10
lr_clean= 1e-3
lr_adv = 5e-4
CLASSES = 10  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"seed = {seed}")