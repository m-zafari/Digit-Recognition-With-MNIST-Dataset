# MLP classifier for digit recognition with Mnist Dataset
This project is implementation of some multi layer perceptron(MLP) neural network consisting of 3 layers fully connected neurons with different activation functions and optimizers(solvers). 
These networks are trained by MNIST Dataset that contains a training set of 60,000 examples, and a test set of 10,000 examples of handwritten digits. 
The tensorflw_HRN.ipynb file is similar to sklrn_HRN.ipynb, so the following text briefly describes the working method of the code written in the sklearn library, and tensorflow is similar, except that tensorflow uses parallel processing.
**First**, we **import** the **required libraries:
1. **Matplotlib.pylot** for plotting graphs
2. **Sklearn.preprocessing** for normalizing data (between 0 and 1)
3. **fetch_openml** for importing the dataset into the written program
4. **sklearn.neural_network.MLPclassifier** for using the structural capabilities of mpl and defining it
5. **numpy** for calculations during network training and...
We import the **mnist728** dataset, which contains 70,000 single-digit images, 10,000 of which are used for testing and the remaining 60,000 for training the network, each image containing 28x28 pixels.
After normalizing the input data of this dataset, then divide it into the two categories mentioned above.
In this project two activations and solvers arrays are used to train all possible networks in order to be able to compare them.
This network consists of 10 inputs and 2 hidden (inner) layers, which have 20 and 50 neurons respectively, and an output layer with 10 external classes.
Finally, we set the requirements for training of mlp networks that includes the alpha coefficient and etc. Then, with the classfier.fit command, the network is fitted on the train data and we create an array called predictions that contains the predicted labels of the x_test_set data. After drawing the accuracy table of all trained methods, we see that the following network has the best convergence:
**classifier = MLPClassifier(
    hidden_layer_sizes=(50,20,10),
    activation = 'relu',  
    solver="sgd",
    alpha=1e-4,
    learning_rate = 'adaptive', 
    learning_rate_init=0.1,
    max_iter=200, 
    random_state=1,
    verbose=10
)**

