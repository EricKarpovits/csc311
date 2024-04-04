import sys

sys.path.append('../')  # Adding the parent directory to the Python path
from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26]

    k_star_by_user = None
    best_val_acc_by_user = 0
    valid_acc_impute_by_user = []
    print("User-Based kNN Collaborative Filtering (Part A):")
    for k in k_values:
        print("Running for k = {}".format(k))
        val_acc = knn_impute_by_user(sparse_matrix, val_data, k)
        valid_acc_impute_by_user.append(val_acc)
        if val_acc > best_val_acc_by_user:
            best_val_acc_by_user = val_acc
            k_star_by_user = k

    test_acc_by_user = knn_impute_by_user(sparse_matrix, test_data, k_star_by_user)
    print("Best k value found is: k* = {} with test accuracy = {}".format(k_star_by_user, test_acc_by_user))

    plt.plot(k_values, valid_acc_impute_by_user)
    plt.title("User-Based Collaborative Filtering")
    plt.suptitle("Validation Accuracy Across Different k Values")
    plt.xlabel("k")
    plt.ylabel("Validation accuracy")
    plt.savefig("./plots/knn/q1_a.pdf")

    k_star_by_item = None
    best_val_acc_by_item = 0
    valid_acc_impute_by_item = []
    print("Item-Based kNN Collaborative Filtering (Part C):")
    for k in k_values:
        print("Running for k = {}".format(k))
        val_acc = knn_impute_by_item(sparse_matrix, val_data, k)
        valid_acc_impute_by_item.append(val_acc)
        if val_acc > best_val_acc_by_item:
            best_val_acc_by_item = val_acc
            k_star_by_item = k
    test_acc_by_item = knn_impute_by_item(sparse_matrix, test_data, k_star_by_item)
    print("Best k value found is: k* = {} with test accuracy = {}".format(k_star_by_item,
                                                                          test_acc_by_item))
    plt.clf() # clear prev plot in memory
    plt.plot(k_values, valid_acc_impute_by_item)

    plt.title("Item-Based Collaborative Filtering")
    plt.suptitle("Validation Accuracy Across Different k Values")
    plt.xlabel("k")
    plt.ylabel("Validation accuracy")
    plt.savefig("./plots/knn/q1_c.pdf")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
