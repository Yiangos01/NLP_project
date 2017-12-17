import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from collections import Counter



binary_targets = False


class Diagnostic_Classifier(nn.Module):
    def __init__(self, hidden_size = 100 ):
        super(Diagnostic_Classifier, self).__init__()
        self.input_size = 500
        self.hidden_size = hidden_size
        self.output_size = 1
        # original weights
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)



    def forward(self, x):

        out = F.relu(self.fc1(x))

        out = F.tanh(self.fc2(out))

        return out

def set_binary_targets(targets):
    # print(type(targets))
    for i in range(len(targets)):
        if targets[i] == 0:
            targets[i] = -1
    return targets

def get_accuracy_wrt_class(targets, predictions):
    class_counter = Counter()
    correct_classifications_counter = Counter()
    for i in range(len(targets)):
        class_counter[targets[i]]+=1
        if targets[i] == predictions[i]:
            correct_classifications_counter[targets[i]]+=1
    accuracies = Counter()
    mean_accuracy = 0
    i =0
    for class_id in class_counter:
        accuracies[class_id] = correct_classifications_counter[class_id] / float(class_counter[class_id])
        mean_accuracy += accuracies[class_id]
        i+=1
    mean_accuracy /= float(i)

    string_result = 'mean accuracy: '+ str(mean_accuracy)
    for class_id in class_counter:
        string_result += ', ' +  str(class_id) + ':' + str(accuracies[class_id]) + ", "
    return accuracies , mean_accuracy , string_result


# def normalize_targets(targets, binary_targets):
#     if binary_targets:
#         for i in range(len(targets)):
#             if targets[i] == 1

#     else:



inputs = np.loadtxt('nsubs_input_big.txt')
# targets = np.loadtxt('nsubs_target_big.txt')

targets = np.loadtxt('train_conj_targets.txt')
# 
# inputs = np.loadtxt('test_nsubs_input.txt')
# targets = np.loadtxt('test_nsubs_targets.txt')


# normalizing targets : left arc -1 -> 0.1 | not nsubj 0 - > 0.5 | right arc 1 -> 0.9
# targets = targets/2.5 + 0.5 # -> so that they have range (0,1)


# counter = np.zeros(3)

# for i, target in enumerate(targets):
#     if target == 1:
#         counter[2] +=1
#     elif target == 0:
#         counter[1] +=1
#     elif target == -1:
#         counter[0] +=1
#     else:
        # print(target)

# print('counters:',counter)
# print('inputs_size:',len(inputs))
# print('targets_size:',len(targets))
# inputs = Variable(torch.from_numpy(inputs)).float()
# targets = Variable(torch.from_numpy(targets)).float()

if binary_targets:
    targets = set_binary_targets(targets)

multiplier = 40

counter = np.zeros(3)
new_targets = []
new_imputs = []
for i, target in enumerate(targets):
    if target == 1:
        for j in range(multiplier):
            new_targets.append(1)
            new_imputs.append(inputs[i])
            counter[2] +=1
    elif target == 0:
        new_targets.append(0)
        new_imputs.append(inputs[i])
        counter[1] +=1
    elif target == -1:
        for j in range(multiplier):
            new_targets.append(-1)
            new_imputs.append(inputs[i])
            counter[0] +=1
    else:
        print(target)


print('counters:',counter)
new_imputs = np.asarray(new_imputs)
new_targets = np.asarray(new_targets)
print('inputs_size:',len(new_imputs))
print('targets_size:',len(new_targets))
inputs = Variable(torch.from_numpy(new_imputs)).float()
targets = Variable(torch.from_numpy(new_targets)).float()



# importing test set


test_inputs = np.loadtxt('test_nsubs_input.txt')

# test_targets = np.loadtxt('test_nsubs_targets.txt')
test_targets = np.loadtxt('test_conj_targets.txt')


if binary_targets:
    test_targets = set_binary_targets(test_targets)


test_inputs = Variable(torch.from_numpy(test_inputs)).float()
test_targets = Variable(torch.from_numpy(test_targets)).float()


# ------------------------------------------------------------------------------------------

# Hyper Parameters
num_epochs = 50
learning_rate = 0.5
net = Diagnostic_Classifier(200)
batch_size = 10

model_name = '200_SGD_MSE_tanh_big_500_epochs_no_dublec'

# enabling cuda
# if torch.cuda.is_available() :
#     inputs = inputs.cuda()
#     targets = targets.cuda()
#     test_inputs = test_inputs.cuda()
#     test_targets = test_targets.cuda()
#     net.cuda()
# else:
#     inputs = inputs.cpu()
#     targets = targets.cpu()
#     test_inputs = test_inputs.cpu()
#     test_targets = test_targets.cpu()
#     net.cpu()


train = data_utils.TensorDataset(inputs.data, targets.data)
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)


# loss_weights = torch.Tensor(3)
# loss_weights[0] = 0.45
# loss_weights[1] = 0.1
# loss_weights[2] = 0.45
criterion = torch.nn.MSELoss()#weight = loss_weights) #, size_average = False)


optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)



train_predictions = np.zeros(inputs.data.size()[0])
test_predictions = np.zeros(test_inputs.data.size()[0])


train_accuracy = np.zeros([num_epochs,4])
test_accuracy = np.zeros([num_epochs,4])


# Train the Model
for epoch in range(1, num_epochs +1):

    # error = 0

    # accuracy = 0

    number_of_batches = 0
    # for input, output in train_loader:
    for temp_input, temp_output in train_loader:

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        temp_input = Variable(temp_input).float()
        temp_output = Variable(temp_output).float()

        # Forward to get output
        out = net(temp_input)
        # print('out:',out[:10])
        # print('temp_output:',temp_output[:10])

        # Calculate Loss
        loss = criterion(out, temp_output)
        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        # number_of_batches+=1
        # error += loss

    # evaluating for train and test sets
    train_out = net(inputs)

    if binary_targets:
        for i in range(train_out.data.size()[0]):
            if  train_out.data[i][0] > 0:
                train_predictions[i] = 1
            else:
                train_predictions[i] = -1
    else:
        for i in range(train_out.data.size()[0]):
            if  train_out.data[i][0] > 0.5:
                train_predictions[i] = 1
            elif train_out.data[i][0] < -0.5:
                train_predictions[i] = -1
            else:
                train_predictions[i] = 0




    train_accuracies , train_mean_accuracy , train_string_result = get_accuracy_wrt_class(targets.data.numpy(), train_predictions)


    test_out = net(test_inputs)
    if binary_targets:
        for i in range(test_inputs.data.size()[0]):
            if  test_out.data[i][0] > 0:
                test_predictions[i] = 1
            else:
                test_predictions[i] = -1
    else:
        for i in range(test_out.data.size()[0]):
            if  test_out.data[i][0] > 0.5:
                test_predictions[i] = 1
            elif test_out.data[i][0] < -0.5:
                test_predictions[i] = -1
            else:
                test_predictions[i] = 0

    # for i in range(test_inputs.data.size()[0]):
    #     if  test_out.data[i][0] > 0:
    #         test_predictions[i] = 1
    #     else:
    #         test_predictions[i] = -1


        # batch_accuracy = 0
        # # update accuracy
        # for i in range(temp_output.data.size()[0]):
        #     if temp_output.data[i] == 1 and out.data[i][0] > 0.5:
        #         train_predictions +=1
        #     elif temp_output.data[i] == -1 and out.data[i][0] < -0.5:
        #         batch_accuracy +=1
        #     elif temp_output.data[i] == 0 and ( out.data[i][0] > -0.5 and out.data[i][0] ) < 0.5:
        #         batch_accuracy +=1
        # batch_accuracy /= temp_output.data.size()[0]

        # accuracy += batch_accuracy

        
    test_accuracies , test_mean_accuracy , test_string_result = get_accuracy_wrt_class(test_targets.data.numpy(), test_predictions)

    # train_predictionstargets = np.ones(train_out.data.size()[0])
    # test_predictions = np.ones(test_out.data.size()[0])
# 
    # accuracy /= number_of_batches

    # if epoch%50 == 0:
    # print(targets.data.numpy().shape, test_targets.data.numpy().shape)
    # torch.save(net.state_dict(), 'Models/' + model_name +'_e_'+ str(epoch) + '.pt')
    train_accuracy[epoch-1][0] = train_mean_accuracy
    train_accuracy[epoch-1][1] = train_accuracies[-1]
    train_accuracy[epoch-1][2] = train_accuracies[0]
    train_accuracy[epoch-1][3] = train_accuracies[1]

    test_accuracy[epoch-1][0] = test_mean_accuracy
    test_accuracy[epoch-1][1] = test_accuracies[-1]
    test_accuracy[epoch-1][2] = test_accuracies[0]
    test_accuracy[epoch-1][3] = test_accuracies[1]

    print('Epoch:', epoch)
    print('Train set :',train_string_result)
    print('Test set  :',test_string_result)
    # print('Train set :',precision_recall_fscore_support(targets.data.numpy(), train_predictions, average='weighted'))
    # print('Test set  :',precision_recall_fscore_support(test_targets.data.numpy(), test_predictions, average='weighted'))
    # After the end of every epoch
    # we print net's output and original output
    # print('original output :',temp_output.data)
    # print('net output :', out.data)

    # we print mean training data
    # error /= number_of_batches
    # print('Epoch :',epoch , ', Accuracy :',accuracy*100, '%, Error:', error.data[0])

np.savetxt('conj_train_accuracies.txt', train_accuracy)
np.savetxt('conj_test_accuracies.txt', test_accuracy)

# we save the model
# net.cpu()
# torch.save(net.state_dict(), 'Models/' + model_name)






