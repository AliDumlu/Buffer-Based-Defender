# Import required libraries
import random
from os import listdir
import numpy as np
from Implementation.Defense import Defense
from keras.preprocessing.image import load_img
from sklearn.utils import shuffle
from statistics import mean
from sklearn.metrics import accuracy_score

attacks = ["FGSM", "BIM", "UAP", "PA", "ST", "ZOO"]
classifiers = ["MLP", "VGG19", "ResNet50"]


def load(attack, classifier):
    path = './Adversarial_Examples/' + attack + '/' + classifier
    org_path = path + '/org/'
    adv_path = path + '/adv/'
    # Get the images files' names
    files_org = listdir(org_path)
    files_adv = listdir(adv_path)
    # Load images into numpy arrays
    x_org = np.array([np.array(load_img(org_path + file)) for file in files_org])
    x_adv = np.array([np.array(load_img(adv_path + file)) for file in files_adv])
    # Normalize
    x_org = x_org.astype('float32')
    x_adv = x_adv.astype('float32')
    x_org = x_org / 255.0
    x_adv = x_adv / 255.0
    return x_org[0:500], x_adv[0:500]


def combine(x_org, x_adv):
    # Create arrays to label original(1) and adversarial(0) images
    length = len(x_org)
    np.random.shuffle(x_org)
    np.random.shuffle(x_adv)
    label_org = np.tile(0, length)
    label_adv = np.tile(1, length)
    # Combine original and adversarial images into one array
    images = np.concatenate((x_org[0:length], x_adv[0:length]))
    # Combine original and adversarial labels into one array
    labels = np.concatenate((label_org, label_adv))
    # Shuffle the images and labels
    # images, labels = shuffle(images, labels, n_samples=length * 2)
    return images, labels


def test(defender, images, labels):
    # Loop over the images to detect adversarial examples
    pred_labels = []
    for image in images:
        if defender.detect(image):
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    pred_labels = np.array(pred_labels)
    # Calculate accuracy
    acc = accuracy_score(labels, pred_labels)
    return acc


def test_buffer_sizes(buffer_sizes, threshold, som_data):
    results_file = open(r"./Results/buffer_size.csv", "w")
    results_file.write("Buffer size," + ",".join(str(x) for x in buffer_sizes))
    for i in range(len(classifiers)):
        accuracies = []
        som_accuracies = []
        results_file.write("\n" + classifiers[i])
        x_org, x_adv = load("PA", classifiers[i])
        images, labels = combine(x_org, x_adv)
        for size in buffer_sizes:
            defender = Defense(n=size, threshold=threshold)
            acc = test(defender, images, labels)
            accuracies.append(acc)
            defender = Defense(n=size, threshold=threshold, som_enabled=True, som_data=som_data)
            acc = test(defender, images, labels)
            som_accuracies.append(acc)
        results_file.write("\nWithout," + ",".join(str(x) for x in accuracies))
        print(accuracies)
        results_file.write("\nWith," + ",".join(str(x) for x in som_accuracies))
        print(som_accuracies)
    results_file.close()


def test_thresholds(thresholds, norm, num_of_runs):
    results_file = open(r"./Results/threshold_L-" + str(norm) + ".csv", "w")
    results_file.write("Threshold," + ",".join(str(x) for x in thresholds))
    accuracies = []
    for i in range(len(classifiers)):
        results_file.write("\n" + classifiers[i] + ",")
        x_org, x_adv = load("PA", classifiers[i])
        for j in range(len(thresholds)):
            for k in range(num_of_runs):
                images, labels = combine(x_org, x_adv)
                defender = Defense(n=1000, threshold=thresholds[j], norm=norm)
                acc = test(defender, images, labels)
                accuracies.append(acc)
            results_file.write(str(mean(accuracies)) + ",")
            accuracies.clear()
    results_file.close()


def test_attacks(threshold, num_of_runs):
    results_file = open(r"./Results/attacks.csv", "w")
    results_file.write("," + ",".join(classifiers))
    accuracies = []
    for i in range(len(attacks)):
        results_file.write("\n" + attacks[i] + ",")
        print("\n" + attacks[i])
        for j in range(len(classifiers)):
            x_org, x_adv = load(attacks[i], classifiers[j])
            for k in range(num_of_runs):
                images, labels = combine(x_org, x_adv)
                defender = Defense(n=1000, threshold=threshold)
                acc = test(defender, images, labels)
                accuracies.append(acc)
            results_file.write(str(mean(accuracies)) + ",")
            accuracies.clear()
    results_file.close()
