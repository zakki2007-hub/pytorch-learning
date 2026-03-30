# -*- coding: gbk -*-
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的是: {device}")