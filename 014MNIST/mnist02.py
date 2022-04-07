import torchvision as tv

source0 = tv.datasets.MNIST("../MNIST", train = True, download = True)
source1 = tv.datasets.MNIST("../MNIST", train = False, download = True)

print(source0)
print(source1)
print()

DATA0 = source0.data
TARGET0 = source0.targets

print(DATA0.shape, DATA0.dtype, DATA0.min(), DATA0.max())
print(TARGET0.shape, TARGET0.dtype, TARGET0.min(), TARGET0.max())
print()