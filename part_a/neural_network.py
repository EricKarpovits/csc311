import sys
sys.path.append('../')  # Adding the parent directory to the Python path
from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt # For plotting


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        h1 = F.sigmoid(self.g(inputs))
        out = F.sigmoid(self.h(h1))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, plot=True):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_cost = []
    valid_accs = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + lamb/2 * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)

        if plot:
            train_cost.append(train_loss)
            valid_accs.append(valid_acc)

        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        
    if plot:
        plt.title("SGD Training Curve")
        plt.plot(train_cost, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Training Cost")
        plt.savefig("./plots/nn/SGDTrainCurve")
        plt.show()

        plt.title("SGD Validation Accuracy")
        plt.plot(valid_accs, label="Valid Acc")
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.savefig("./plots/nn/SGDValidAcc")
        plt.show()

    return valid_acc

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)



def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################


    ##### Hyperparameter Tuning: Most Performant (k, lr, num_epoch) = (50, 0.05, 10) ######
    # lamb = 0.2
    # k_list = [10,50,100,200,500]

    # lr_list = [0.01, 0.05, 0.005]
    # num_epoch_list = [10, 20, 40, 60]

    # tune_iter = 1
    # tuned_params = [0, 0, 0, 0, 0] # [k, lr, num_epoch, valid_acc]
    # param_combs = len(k_list) * len(lr_list) * len(num_epoch_list)
    # for curr_k in k_list:
    #     for curr_lr in lr_list:
    #         for curr_epoch in num_epoch_list:
    #             print("############### Trial {}/{}: k = {}, lr = {}, num_epoch = {} ###############".format(tune_iter, param_combs, curr_k, curr_lr, curr_epoch))

    #             curr_model = AutoEncoder(train_matrix.shape[1], curr_k)
    #             curr_valid_acc =  train(curr_model, curr_lr, lamb, train_matrix, zero_train_matrix, valid_data, curr_epoch, plot=False)

    #             if curr_valid_acc > tuned_params[-1]:
    #                 tuned_params = [curr_k, curr_lr, curr_epoch, curr_valid_acc]

    #             print("best_params = " + str(tuned_params))    
    #             tune_iter += 1

    # print("FINISHED!!!!!!!!!!!!" + str(tuned_params))

    
    ##### Plotting Optimal hyperparams: Test Accuracy = 0.6782387806943269 #####
    k = 50
    model = AutoEncoder(train_matrix.shape[1], k)
    # Set optimization hyperparameters.
    lr = 0.05
    num_epoch = 10
    lamb = 0
    # train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    # test_acc = evaluate(model, zero_train_matrix, test_data)
    # print(test_acc)


    ###### Tuning Regularizer weight: Most Performant Lambda = 0.001 ###### 
    # lamb_list = [0.001, 0.01, 0.1, 1]
    # tuned_lamb = [0, 0]     # [lambda, valid_acc]
    # for curr_lamb in lamb_list:
    #     print("########### Lambda = {} ###########".format(curr_lamb))
    #     curr_model = AutoEncoder(train_matrix.shape[1], k)
    #     curr_valid_acc =  train(curr_model, lr, curr_lamb, train_matrix, zero_train_matrix, valid_data, num_epoch, plot=False)

    #     if curr_valid_acc > tuned_lamb[-1]:
    #         tuned_lamb = [curr_lamb, curr_valid_acc]
        
    # print("Best Lambda = " + str(tuned_lamb[0]))

    lamb = 0.001
    train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    test_acc = evaluate(model, zero_train_matrix, test_data)
    print(test_acc)
 
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
