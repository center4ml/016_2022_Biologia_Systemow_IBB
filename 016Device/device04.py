import torch

tensor1 = torch.tensor([7., 3., 5.])
print(tensor1.device)

print(torch.cuda.is_available())

tensor2 = tensor1.to("cuda")
print(tensor2.device)
