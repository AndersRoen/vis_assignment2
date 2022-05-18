# vis_assignment_2
Link to github repo: https://github.com/AndersRoen/vis_assignment2.git

## Assignment 2 description
In the last two classes, you've seen how to train a model that can be used to make classification predictions on image data. So far, we've seen two different approaches. The first approach is a simple logistic regression classifier; the second uses a (very inefficient!) neural network class written in numpy.

In class, we saw how these classifiers worked through the use of Notebooks in Jupyter Lab. However, as we spoke about in class, these Notebooks are not the best way to use and share code. Instead, what we want is a script that can be run and which then produces the outputs with minimal input from the user.

For this assignment, you will take the classifier pipelines we covered in lecture 7 and turn them into two separate .py scripts. Your code should do the following:

    One script should be called logistic_regression.py and should do the following:
        Load either the MNIST_784 data or the CIFAR_10 data
        Train a Logistic Regression model using scikit-learn
        Print the classification report to the terminal and save the classification report to out/lr_report.txt
    Another scripts should be called nn_classifier.py and should do the following:
        Load either the MNIST_784 data or the CIFAR_10 data
        Train a Neural Network model using the premade module in neuralnetwork.py
        Print output to the terminal during training showing epochs and loss
        Print the classification report to the terminal and save the classification report to out/nn_report.txt
## Bonus tasks

    Use argparse() so that the scripts use either MNIST_784 or CIFAR_10 based on some input from the user on the command line
    Use argparse() to allow users to define the number and size of the layers in the neural network classifier.
    Write the script in such a way that it can take either MNIST_784 or CIFAR_10 or any data that the user wants to classify
        You can determine how the user data should be structured, by saying that it already has to be pre-processed or feature extracted.

## Methods
This assignment has two scripts: ```logistic_regression.py``` and ```nn_classifier.py```.
To get this to run optimally, you should run the ```setup.sh``` which installs necessary dependencies.
This assignment is about image classification, and which kind of classifier yields the best results. The scripts generally do very similar things, both allow the user to choose between two datasets: ```cifar10``` or ```mnist_784``` by using the command line argument ```--data```. If the user types in ```mnist_784``` it will run that dataset, if the user types in anything else, it will run ```cifar10```. 
The scripts will then load in the chosen dataset, do a ```train_test_split```, normalise, reshape. The ```cifar10``` dataset will also be converted to greyscale, which isn't necessary for ```mnist_784``` as it is already black and white. 
The scripts only really differ when it comes to their classifiers. ```logistic_regression.py``` uses a simple logistic regression classifier, which is fitted to the training data and then makes predictions, and ```nn_classifier.py``` creates a neural network from which it can generate predictions. I've set the training to 10 epochs. Both scripts generates a classification report with a user-defined name. The ```nn_classifier.py``` script will have a user-defined layer size.

## Usage
```logistic_regression.py``` has two required command line arguments: ```--data``` which defines which dataset you want and ```--rep_name``` which defines the filename of the classification report. To run the script point the command line to the ```vis_assignment2``` folder and include the required arguments.
```nn_classifier.py``` has three required command line arguments: ```--data``` which defines which dataset you want, ```--rep_name``` which defines the filename of the classification report and ```--layer``` which defines the layer size of the neural network. I've worked with a layer size of 64.

## Results
Generally, the neural network performs better than the logistic regression classifier and ```mnist_784``` yields better results than ```cifar10```. This is probably due to the fact that ```cifar10``` is a significantly more complex dataset, with real life images and ```mnist_784``` is black and white images of numbers, with nothing else. 
