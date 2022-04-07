from sklearn import datasets

source = datasets.load_iris()
data = source.data
target = source.target

print(data)
