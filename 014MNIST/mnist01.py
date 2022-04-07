import torchvision as tv

source0 = tv.datasets.MNIST("../MNIST", train = True, download = True)
source1 = tv.datasets.MNIST("../MNIST", train = False, download = True)

print(source0)
print(source1)
print()
