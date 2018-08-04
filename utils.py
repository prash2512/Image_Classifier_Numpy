import numpy as np
import matplotlib.pyplot as plt
from classifiers import KNearestNeighbor

def plot_samples(X_train,y_train):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

def generate_folds(X_train,y_train,num_folds):
    X_train_folds = np.split(X_train,num_folds)
    y_train_folds = np.split(y_train,num_folds)
    return X_train_folds,y_train_folds


def run_k_fold_cross_validation(X_train,y_train,num_folds,k,k_accuracy):
    X_train_folds,y_train_folds = generate_folds(X_train,y_train,num_folds)
    accuracy = 0.0
    accuracy_list = []
    for i in range(num_folds):
        val_fold_x = X_train_folds[i]
        val_fold_y = y_train_folds[i]
        temp_X_train = np.concatenate(X_train_folds[:i] + X_train_folds[i+ 1:])
        temp_y_train = np.concatenate(y_train_folds[:i] + y_train_folds[i + 1:])
        classifier = KNearestNeighbor()
        classifier.train(temp_X_train,temp_y_train)
        dists = classifier.compute_distances_no_loops(val_fold_x)
        val_pred_y = classifier.predict_labels(dists,k)
        num_correct = np.sum(val_pred_y == val_fold_y)
        accuracy_list.append((float(num_correct) / val_pred_y.shape[0]))
        accuracy = accuracy+(float(num_correct) / val_pred_y.shape[0])
    k_accuracy[k] = accuracy_list
    accuracy = accuracy/num_folds
    return accuracy


        
def choose_best_k(X_train,y_train,num_folds):
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
    k_accuracy = {}
    max_accuracy = 0.0
    max_accuracy_k = 0
    for k in k_choices:
        accuracy = run_k_fold_cross_validation(X_train,y_train,num_folds,k,k_accuracy)
        if accuracy>max_accuracy:
            max_accuracy = accuracy
            max_accuracy_k = k
    plot_cross_validation_accuracy(k_choices,k_accuracy)
    return max_accuracy_k 

def interpolate(dists):
    plt.imshow(dists, interpolation='none')
    plt.show()

def plot_cross_validation_accuracy(k_choices,k_to_accuracies):
    # plot the raw observations
    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()