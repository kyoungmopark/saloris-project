import torch
ckpt = torch.load("C:/cctvproject/New/best.pt")  # applies to both official and custom models
torch.save(ckpt, "updated-best.pt")