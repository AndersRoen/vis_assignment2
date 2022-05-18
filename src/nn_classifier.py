# path tools
import sys,os

import argparse
# image processing
import cv2

# neural networks with numpy
import numpy as np
from tensorflow.keras.datasets import cifar10 #pip install tensorflow
from utils.neuralnetwork import NeuralNetwork
import pandas as pd

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

def choose(data):
    dataset = data
    return dataset

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
    #converting to arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    
    y_train = LabelBinarizer().fit_transform(y_train) #binarizing labels
    y_test = LabelBinarizer().fit_transform(y_test)
    


    #turning it greyscale
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

    # normalizing stuff
    X_train_scaled = X_train_grey/255
    X_test_scaled = X_test_grey/255

    # reshaping data
    nsamples, nx, ny = X_train_scaled.shape
    X_train_scaled = X_train_scaled.reshape((nsamples, nx*ny))
    nsamples, nx, ny = X_test_scaled.shape
    X_test_scaled = X_test_scaled.reshape((nsamples, nx*ny))
    return X_train_scaled, X_test_scaled, y_train, y_test



# if you chose mnist_784
def load_process_mnist(dataset):
    X, y = fetch_openml(dataset, return_X_y = True)
    X = np.array(X)
    y = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    random_state = 9,
                                                    train_size = 7500, 
                                                    test_size = 2500)

    y_train = LabelBinarizer().fit_transform(y_train) #binarizing labels
    y_test = LabelBinarizer().fit_transform(y_test)

    X_train_scaled = np.array(X_train/255) #normalizing
    X_test_scaled = np.array(X_test/255)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
    
def nn_model(X_train_scaled, layer, y_train):
    input_shape = X_train_scaled.shape[1]
    nn = NeuralNetwork([input_shape, int(layer), 10]) 
    nn.fit(X_train_scaled, y_train,  epochs = 10, displayUpdate=1) 
    return nn

def predictor(nn, X_test_scaled, y_test, rep_name):
    predictions = nn.predict(X_test_scaled)
    y_pred = predictions.argmax(axis=1)
    report = classification_report(y_test.argmax(axis=1), y_pred, output_dict=True)
    root = "out"
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(root, rep_name))
    return report_df

def parse_args():
    # initialize argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True, help="the dataset to use")
    ap.add_argument("-l", "--layer", required = True, help = "the desired layer size of the model")
    ap.add_argument("-rp", "--rep_name", required=True, help="the name of the classification report")
    args = vars(ap.parse_args())
    return args

def main():
    args = parse_args()
    dataset = choose(args["data"])
    # if the user wants mnist_784
    if dataset == "mnist_784":
        X_train_scaled, X_test_scaled, y_train, y_test = load_process_mnist(dataset)
        nn = nn_model(X_train_scaled, args["layer"], y_train)
        report_df = predictor(nn, X_test_scaled, y_test, args["rep_name"])
    # if it's anything else
    else:
        X_train_scaled, X_test_scaled, y_train, y_test = load_process_cifar10()
        nn = nn_model(X_train_scaled, y_train)
        report_df = predictor(nn, X_test_scaled, y_test, args["rep_name"])

if __name__ == "__main__":
    main()
