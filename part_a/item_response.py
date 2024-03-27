from utils import *

import numpy as np

import matplotlib.pyplot as plt

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))

def process_data(data, theta, beta):
    """ From the given data, obtain the np array of user_id, question_id,
    is_correct, theta values corresponding to the user_id values, and
    beta values corresponding to the question_id values

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: np arrays
    """
    user_id = np.array(data['user_id'])
    question_id = np.array(data['question_id'])
    is_correct = np.array(data['is_correct'])
    data_theta = theta[user_id]
    data_beta = beta[question_id]
    return user_id, question_id, is_correct, data_theta, data_beta

def index_process_data(is_correct, data_theta, data_beta, data_index):
    """ Get the np arrays of the is_correct values, data_theta values and
    data_beta values at each index value in data_index array

    :param is_correct: np array
    :param data_theta: np array
    :param data_beta: np array
    :param data_index: np array
    :return: np arrays
    """
    return is_correct[data_index], data_theta[data_index], data_beta[data_index]

def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    user_id, question_id, is_correct, data_theta, data_beta = process_data(data, theta, beta)

    log_lklihood = np.sum(is_correct*(data_theta - data_beta) - (np.log(np.ones(len(user_id))+np.exp(data_theta - data_beta))))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_id, question_id, is_correct, data_theta, data_beta = process_data(data, theta, beta)
    new_theta = []
    for theta_index in range(542):
        data_index = np.where(user_id == theta_index)[0]
        curr_is_correct, curr_data_theta, curr_data_beta = index_process_data(is_correct, data_theta, data_beta, data_index)
        theta_beta_difference = curr_data_theta - curr_data_beta
        new_theta.append(np.sum(curr_is_correct - sigmoid(theta_beta_difference)))
    new_beta = []
    for beta_index in range(1774):
        data_index = np.where(question_id == beta_index)[0]
        curr_is_correct, curr_data_theta, curr_data_beta = index_process_data(is_correct, data_theta, data_beta, data_index)
        theta_beta_difference = curr_data_theta - curr_data_beta
        new_beta.append(np.sum(sigmoid(theta_beta_difference) - curr_is_correct))
    theta = theta + lr * np.array(new_theta)
    beta = beta + lr * np.array(new_beta)
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(542)
    beta = np.zeros(1774)

    val_acc_lst = []
    training_log_likelihood = []
    valid_log_likelihood = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        training_log_likelihood.append(neg_lld)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        valid_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        valid_log_likelihood.append(valid_neg_lld)
        theta, beta = update_theta_beta(data, lr, theta, beta)

    plt.title("Negative Log likelihood for Train Set")
    plt.plot([x for x in range(iterations)], training_log_likelihood)
    plt.xlabel("Number of iterations")
    plt.ylabel("Negative log likelihood")
    plt.show()

    plt.title("Negative Log likelihood for Validation Set")
    plt.plot([x for x in range(iterations)], valid_log_likelihood)
    plt.xlabel("Number of iterations")
    plt.ylabel("Negative log likelihood")
    plt.show()
    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    itr = 50
    lr = 0.00271
    theta, beta, val_ac_lst = irt(data=train_data, val_data=val_data, lr=lr, iterations=itr)
    print("Validation Accuracy:" + str(evaluate(data=val_data, theta=theta, beta=beta)))
    print("Testing Accuracy:" + str(evaluate(data=test_data, theta=theta, beta=beta)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################

    # I will pick Q1, Q100, Q500
    QID = [1, 100, 500]
    total_likelihood = []
    sorted_theta = sorted(theta)
    for each_qid in QID:
        likelihood = []
        beta_value = beta[each_qid]
        for each_theta_value in sorted_theta:
            likelihood.append(sigmoid(each_theta_value - beta_value))
        total_likelihood.append(likelihood)

    plt.title("Likelihood of corretly answering each question with respect to theta")
    plt.plot(sorted_theta, total_likelihood[0], label="Q1 likelihood of answering correctly")
    plt.plot(sorted_theta, total_likelihood[1], label="Q100 likelihood of answering correctly")
    plt.plot(sorted_theta, total_likelihood[2], label="Q500 likelihood of answering correctly")
    plt.xlabel("Theta")
    plt.ylabel("Likelihood of getting correct answer")
    plt.legend()
    plt.show()

    # Draft Written Answer: TODO: DELETE ME after typing into the document
    # The curves generally all go upward in S shape with increasing theta value, with relatively lower slope on the 2 ends and higher slope in the middle.
    # These curves represent how the students' abilities affect their chances of answering each question correctly
    # This is reflecting the difficulty level of each question as well because the ones that are generally steeper would have lower difficulty because
    # students can largely improve their chances of correctly answering them with slight increase of their abilities, while those with
    # lower slope values will requires higher improvement on students' abilities to increase students' chances of correctly answering them.

    # #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
