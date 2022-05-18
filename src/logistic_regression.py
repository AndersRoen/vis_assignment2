# path tools
import sys,os
sys.path.append(os.path.join("utils"))
import pandas as pd
import argparse
# image processing
import cv2

# neural networks with numpy
import numpy as np
from tensorflow.keras.datasets import cifar10 #pip install tensorflow
#from neuralnetwork import NeuralNetwork

# machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# a function that allows that user to choose which dataset they want to work with - either "mnist_784" ot "cifar10"
def choose(data):
    dataset = data
    return dataset

# if the user want cifar10

def load_process_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    labels = ["airplane",
          "automobile",
          "bird",
          "cat",
          "deer",
          "dog",
          "frog",
          "horse",
          "ship",
          "truck"]

    #turning it greyscale
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

    # normalizing stuff
    X_train_scaled = (X_train_grey - X_train_grey.min())/(X_train_grey.max() - X_train_grey.min())
    X_test_scaled = (X_test_grey - X_test_grey.min())/(X_test_grey.max() - X_test_grey.min())

    # reshaping data
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples, nx*ny))
    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples, nx*ny))
    return X_train_dataset, X_test_dataset, y_train, y_test

#if the user wants mnist
def load_process_mnist(dataset):
    # load the data
    X, y = fetch_openml(dataset, return_X_y = True)
    # convert to arrays
    X = np.array(X)
    y = np.array(y)
    # create train-test-splot
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        random_state = 42,
                                                        train_size = 7500) 
   
    # normalize
    X_train_scaled = X_train/255
    X_test_scaled = X_test/255
    return X_train_scaled, X_test_scaled, y_train, y_test

def classifier(X_train_dataset, y_train, X_test_dataset, y_test, rep_name):
    clf = LogisticRegression(penalty = "none",
                         tol = 0.1,
                         solver = "saga",
                         multi_class = "multinomial").fit(X_train_dataset, y_train)

    y_pred = clf.predict(X_test_dataset)
    report = classification_report(y_test, y_pred, output_dict = True)
    root = "out"
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(root, rep_name))
    return report_df

def parse_args():
    # initialize argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True, help="the dataset to use")
    ap.add_argument("-rp", "--rep_name", required=True, help="the name of the classification report")
    args = vars(ap.parse_args())
    return args
    

def main():
    args = parse_args()
    dataset = choose(args["data"])
    # if the user wants mnist_784
    if dataset == "mnist_784":
        X_train_scaled, X_test_scaled, y_train, y_test = load_process_mnist(dataset)
        report_df = classifier(X_train_scaled, y_train, X_test_scaled, y_test, args["rep_name"])
    # if it is anything else
    else:
        X_train_scaled, X_test_scaled, y_train, y_test = load_process_cifar10()
        report_df = classifier(X_train_scaled, y_train, X_test_scaled, y_test, args["rep_name"])
    
if __name__ == "__main__":
    main()
    






    
    

