import torch

tensor1 = torch.tensor([7., 3., 5.])
print(tensor1.device)

print(torch.cuda.is_available())

tensor2 = tensor1.to("cuda")
print(tensor1)
print(tensor2)
print(tensor2.device)

tensor3 = tensor1.cuda()
print(tensor3)

tensor4 = torch.tensor([7., 3., 5.], device = "cuda")
print(tensor4)
