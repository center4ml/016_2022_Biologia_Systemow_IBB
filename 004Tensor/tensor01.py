from sklearn import datasets

features = 4
classes = 3

source = datasets.load_iris()
data = source.data
target = source.target

print(data)
print(target)