import numpy as np
import math
import matplotlib.pyplot as plt
from classifiers import KNearestNeighbor
from classifiers.linear_classifier import LinearSVM
from gradient_check import grad_check_sparse

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

def subtract_mean_image(X_train,X_val,X_test,X_dev):
    # second: subtract the mean image from train and test data
    mean_image = get_mean_image(X_train)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image
    return X_train,X_val,X_test,X_dev

def get_mean_image(X_train):
    # Preprocessing: subtract the mean image
    # first: compute the image mean based on the training data
    mean_image = np.mean(X_train, axis=0)
    return mean_image

def plot_show_meaniamge(mean_image):
    print(mean_image[:10]) # print a few of the elements
    plt.figure(figsize=(4,4))
    plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image
    plt.show()

def plot_loss(loss_hist):
    # A useful debugging strategy is to plot the loss as a function of
    # iteration number:
    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()

def choose_best_svm(X_train,y_train,X_val,y_val,learning_rates,regularization_strengths):
    results = {}
    best_svm = -1
    best_val = None
    for lr in learning_rates:
        for reg in regularization_strengths:
            svm = LinearSVM()
            loss_hist = svm.train(X_train,y_train,learning_rate=lr,reg=reg,num_iters=1000)
            y_train_pred = svm.predict(X_train)
            print('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))
            y_val_pred = svm.predict(X_val)
            print('validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))
            results[(lr,reg)] = (np.mean(y_train == y_train_pred),np.mean(y_val == y_val_pred),svm)
            for lr, reg in sorted(results):
                train_accuracy, val_accuracy,svm = results[(lr, reg)]
                if val_accuracy>best_val:
                    best_svm = svm
                    best_val = val_accuracy
                print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                            lr, reg, train_accuracy, val_accuracy))
    return results,best_svm,best_val

def plot_cross_validation_svm(results):
    # Visualize the cross-validation results
    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]

    # plot training accuracy
    marker_size = 100
    colors = [results[x][0] for x in results]
    plt.subplot(2, 1, 1)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 training accuracy')

    # plot validation accuracy
    colors = [results[x][1] for x in results] # default size of markers is 20
    plt.subplot(2, 1, 2)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 validation accuracy')
    plt.show()

def visualize_bestsvm_weights(best_svm):
    # Visualize the learned weights for each class.
    # Depending on your choice of learning rate and regularization strength, these may
    # or may not be nice to look at.
    w = best_svm.W[:-1,:] # strip out the bias
    w = w.reshape(32, 32, 3, 10)
    w_min, w_max = np.min(w), np.max(w)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        
        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])

    plt.show()