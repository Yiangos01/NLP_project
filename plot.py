import numpy as np
import matplotlib.pyplot as plt

x = np.arange(50);

# nsubj

nsubj_train_accuracies = np.loadtxt('accuracies/nsubj_train_accuracies.txt')
nsubj_test_accuracies = np.loadtxt('accuracies/nsubj_test_accuracies.txt')

plt.subplot(2, 1, 1)
plt.plot(x, nsubj_train_accuracies[:,0],x, nsubj_train_accuracies[:,1],x, nsubj_train_accuracies[:,2],x, nsubj_train_accuracies[:,3])
# plt.title('nsubj Dependency Classification Accuracies for Training (Upper) and Test (Lower) Sets.')
plt.ylabel('Accuracy')
plt.xlabel('Epochs of training')
plt.legend(['Mean Accuracy', 'Left Arc Dependencies', 'Not Involved Words', 'Right Arc Dependencies'])

plt.subplot(2, 1, 2)
plt.plot(x, nsubj_test_accuracies[:,0],x, nsubj_test_accuracies[:,1],x, nsubj_test_accuracies[:,2],x, nsubj_test_accuracies[:,3])
plt.ylabel('Accuracy')
plt.xlabel('Epochs of training')
plt.legend(['Mean Accuracy', 'Left Arc Dependencies', 'Not Involved Words', 'Right Arc Dependencies'])

plt.show()



# conj

# conj_train_accuracies = np.loadtxt('accuracies/conj_train_accuracies.txt')
# conj_test_accuracies = np.loadtxt('accuracies/conj_test_accuracies.txt')

# plt.subplot(2, 1, 1)
# plt.plot(x, conj_train_accuracies[:,0],x, conj_train_accuracies[:,1],x, conj_train_accuracies[:,2],x, conj_train_accuracies[:,3])
# # plt.title('conj Dependency Classification Accuracies for Training (Upper) and Test (Lower) Sets.')
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs of training')
# plt.legend(['Mean Accuracy', 'Left Arc Dependencies', 'Not Involved Words', 'Right Arc Dependencies'])

# plt.subplot(2, 1, 2)
# plt.plot(x, conj_test_accuracies[:,0],x, conj_test_accuracies[:,1],x, conj_test_accuracies[:,2],x, conj_test_accuracies[:,3])
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs of training')
# plt.legend(['Mean Accuracy', 'Left Arc Dependencies', 'Not Involved Words', 'Right Arc Dependencies'])

# plt.show()

