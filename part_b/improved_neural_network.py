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

import matplotlib.pyplot as plt  # For plotting


def load_meta_data(path):
    data = np.genfromtxt(path, delimiter=',', usecols=(0, 1, 3), skip_header=1)

    data_sorted = data[data[:, 0].argsort()]
    return np.delete(data_sorted, 0, axis=1)


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

    metaData = load_meta_data("../data/student_meta.csv")
    # Adding the student metadata to the train matrix (gender + premium pupil)

    train_matrix = np.concatenate((train_matrix, metaData), axis=1)
    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0

    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, encoder_hidden_size, bottleneck_size):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, encoder_hidden_size)
        self.encode = nn.Linear(encoder_hidden_size, bottleneck_size)
        self.decode = nn.Linear(bottleneck_size, encoder_hidden_size)
        self.h = nn.Linear(encoder_hidden_size, num_question)

        self.dropout = nn.Dropout(0.25)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        encode_w_norm = torch.norm(self.encode.weight, 2) ** 2
        decode_w_norm = torch.norm(self.decode.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + encode_w_norm + decode_w_norm + h_w_norm

    def get_L1_regularizer_term(self):
        """ Return absolute_value(||W^1||) + absolute_value(||W^2||).

        :return: float
        """
        g_w_norm = torch.abs(torch.norm(self.g.weight, 2))
        encode_w_norm = torch.abs(torch.norm(self.encode.weight, 2))
        decode_w_norm = torch.abs(torch.norm(self.decode.weight, 2))
        h_w_norm = torch.abs(torch.norm(self.h.weight, 2))
        return g_w_norm + encode_w_norm + decode_w_norm + h_w_norm

    def forward(self, inputs, reg=None):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        if reg == "Dropout":
            h1 = nn.ReLU()(self.g(inputs))
            h1_dropout = self.dropout(h1)
            h2 = nn.Sigmoid()(self.encode(h1_dropout))
            h2_dropout = self.dropout(h2)
            out = nn.Sigmoid()(self.h(self.decode(h2_dropout)))
        else:
            h1 = nn.ReLU()(self.g(inputs))
            h2 = nn.Sigmoid()(self.encode(h1))
            out = nn.Sigmoid()(self.h(self.decode(h2)))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out

def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, plot=True, train_data_csv=None, reg=None):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :param train_data_csv: Dict
    :param reg: str
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
    train_accs = []

    highest_valid_acc = 0.0
    epoch_at_highest_acc = 0.0

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs, reg)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            if reg == "L2":
                reg_value = model.get_weight_norm()
            elif reg == "L1":
                reg_value = model.get_L1_regularizer_term()
            else:
                reg_value = 0

            loss = torch.sum((output - target) ** 2.) + lamb / 2 * reg_value
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        if train_data_csv:
            train_acc = evaluate(model, zero_train_data, train_data_csv)

        if valid_acc > highest_valid_acc:
            highest_valid_acc = valid_acc
            epoch_at_highest_acc = epoch

        if plot:
            train_cost.append(train_loss)
            valid_accs.append(valid_acc)
            if train_data_csv:
                train_accs.append(train_acc)
        if train_data_csv:
            print("Epoch: {} \tTraining Cost: {:.6f}\t "
                  "Valid Acc: {}\t Training Acc: {}".format(epoch, train_loss, valid_acc, train_acc))
        else:
            print("Epoch: {} \tTraining Cost: {:.6f}\t "
                  "Valid Acc: {}".format(epoch, train_loss, valid_acc))

    if plot:
        reg_type = reg if reg is not None else "No"
        plt.title("SGD Training Curve with {} Regularizer".format(reg_type))
        plt.plot(train_cost, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Training Cost")
        plt.savefig("./plots/nn/reg/SGDTrainCurveWith{}Regularizer".format(reg_type))
        plt.show()

        if train_data_csv:
            plt.title("SGD Training Accuracy with {} Regularizer".format(reg_type))
            plt.plot(train_accs, label="Training Acc")
            plt.xlabel("Epochs")
            plt.ylabel("Training Accuracy")
            plt.savefig("./plots/nn/reg/SGDTrainingAccWith{}Regularizer".format(reg_type))
            plt.show()

        plt.title("SGD Validation Accuracy with {} Regularizer".format(reg_type))
        plt.plot(valid_accs, label="Valid Acc")
        plt.scatter(epoch_at_highest_acc, highest_valid_acc, color='red')
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.savefig("./plots/nn/reg/SGDValidAccWith{}Regularizer".format(reg_type))
        plt.show()

    return valid_acc, valid_accs, train_accs, train_cost

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

    ##### Hyperparameter Tuning: Most Performant (encoder_k, bottleneck_k, lr, num_epoch) = (50, 10, 0.001, 150) ######
    # lamb = 0
    # since we know that the dimensions have to decrease the first hidden layer should be larger than the second hidden layer
    # encoder_hidden_size = [200, 150, 100, 50]
    # bottleneck_size = [30, 20, 10, 5]

    # lr_list = [0.1, 0.05, 0.001]
    # num_epoch_list = [10, 30, 50, 100, 150, 200]

    # tune_iter = 1
    # tuned_params = [0, 0, 0, 0, 0] # [encoder_dim, bottleneck_dim, lr, num_epoch, valid_acc]
    # param_combs = len(encoder_hidden_size) * len(lr_list) * len(num_epoch_list) * len(bottleneck_size)
    # for encoder_dim in encoder_hidden_size:
    #     for bottleneck_dim in bottleneck_size:
    #         for curr_lr in lr_list:
    #             for curr_epoch in num_epoch_list:
    #                 print("############### Trial {}/{}: encoder_dim = {}, bottleneck_dim = {}, lr = {}, num_epoch = {} ###############".format(tune_iter, param_combs, encoder_dim, bottleneck_dim, curr_lr, curr_epoch))

    #                 curr_model = AutoEncoder(train_matrix.shape[1], encoder_dim, bottleneck_dim)
    #                 curr_valid_acc, valid_accs, train_accs, train_cost = train(curr_model, curr_lr, lamb, train_matrix, zero_train_matrix, valid_data, curr_epoch, plot=False)

    #                 if curr_valid_acc > tuned_params[-1]:
    #                     tuned_params = [encoder_dim, bottleneck_dim, curr_lr, curr_epoch, curr_valid_acc]

    #                 print("best_params = " + str(tuned_params))
    #                 tune_iter += 1

    # print("FINISHED!!!!!!!!!!!!" + str(tuned_params))

    #### Plotting Optimal hyperparams: Valid Accuracy = 0.7066045723962744 #####
    encoder_dim = 50
    bottleneck_dim = 10
    model = AutoEncoder(train_matrix.shape[1], encoder_dim, bottleneck_dim)
    # Set optimization hyperparameters.
    lr = 0.001
    num_epoch = 150
    lamb = 0
    valid_acc, valid_accs, train_accs, train_cost = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    print(valid_acc)


    ###### Tuning Regularizer weight: Most Performant Lambda = 0 ######
    # lamb_list = [0, 0.001, 0.01, 0.1, 1]
    # num_epoch_list = [10, 30, 50, 100, 150, 200]
    # tuned_lamb = [0, 0, 0]     # [lambda, epochs, valid_acc]
    # for epoch in num_epoch_list: #  Not sure if I should rerun it for all epochs... prob just keep epoch = 150
    #     for curr_lamb in lamb_list:
    #         print("########### Lambda = {}, epoch = {} ###########".format(curr_lamb, epoch))
    #         curr_model = AutoEncoder(train_matrix.shape[1], encoder_dim, bottleneck_dim)
    #         curr_valid_acc, valid_accs, train_accs, train_cost =  train(curr_model, lr, curr_lamb, train_matrix, zero_train_matrix, valid_data, epoch, plot=False)

    #         if curr_valid_acc > tuned_lamb[-1]:
    #             tuned_lamb = [curr_lamb, epoch, curr_valid_acc]

    #         print("best_params = " + str(tuned_lamb))

    # print("Best Lambda = {}, with valid acc = {}, epoch = {}".format(tuned_lamb[0], tuned_lamb[2], tuned_lamb[1]))

    # lamb = 0
    # train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    # test_acc = evaluate(model, zero_train_matrix, test_data)
    # print(test_acc)



    ##### Compare different chocies of regularizer #####
    regularizer = [None, "L2", "L1", "Dropout"]
    train_data_csv = load_train_csv("../data")
    different_regularizer_valid_accs_lst = []
    different_regularizer_train_accs_lst = []

    lamb = 0.001
    print("Training with encoder_dim = {}, bottleneck_dim = {}, lr = {}, num_epoch = {}, lamb = {}".format(encoder_dim, bottleneck_dim, lr, num_epoch, lamb))
    for each_regularizer in regularizer:
        print("Current Regularizer: " + str(each_regularizer))
        model = AutoEncoder(train_matrix.shape[1], encoder_dim, bottleneck_dim)
        valid_acc, valid_accs, train_accs, train_cost = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch, train_data_csv=train_data_csv, reg=each_regularizer)
        different_regularizer_train_accs_lst.append(train_accs)
        different_regularizer_valid_accs_lst.append(valid_accs)
        print(valid_acc)

    plt.title("SGD training accuracies of different regularizers")
    plt.plot(different_regularizer_train_accs_lst[0], label="Without Regularizer")
    plt.plot(different_regularizer_train_accs_lst[1], label="L2 Regularizer")
    plt.plot(different_regularizer_train_accs_lst[2], label="L1 Regularizer")
    plt.plot(different_regularizer_train_accs_lst[3], label="Dropout Regularizer")
    plt.xlabel("Epoches")
    plt.ylabel("Training Accuracy")
    plt.legend()
    plt.savefig("./plots/nn/reg/training accuracy of different regularizer")
    plt.show()

    plt.title("SGD validation accuracies of different regularizers")
    plt.plot(different_regularizer_valid_accs_lst[0], label="No Reg")
    plt.plot(different_regularizer_valid_accs_lst[1], label="L2 Reg")
    plt.plot(different_regularizer_valid_accs_lst[2], label="L1 Reg")
    plt.plot(different_regularizer_valid_accs_lst[3], label="Dropout Reg")
    plt.xlabel("Epoches")
    plt.ylabel("Validating Accuracy")
    plt.legend()
    plt.savefig("./plots/nn/reg/validation accuracy of different regularizers")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
