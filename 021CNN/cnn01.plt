from matplotlib import pyplot as plt
import numpy as np

history = np.loadtxt('cnn01.txt')

plt.figure(figsize = (6, 6), tight_layout = True)

plt.subplot(2, 1, 1)
plt.plot(history[:, 1])
plt.plot(history[:, 3])
plt.grid()
plt.ylim(0., 0.5)
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(history[:, 2])
plt.plot(history[:, 4])
plt.grid()
plt.ylim(0.9, 1.)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

epoch = history[:, 4].argmax()
print(epoch, history[epoch, 4])

plt.show()
