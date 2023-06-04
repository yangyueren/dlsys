import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    images = []
    with gzip.open(image_filename, 'rb') as f:
        file_content = f.read()
        offset = 0
        magic_num, num = struct.unpack_from('>ii', file_content, offset=offset)
        offset += 8
        assert magic_num == 2051, 'train image error'
        rows, cols = struct.unpack_from('>ii', file_content, offset=offset)
        offset += 8
        for _ in range(num):
            image = []
            for i in range(rows):
                for j in range(cols):
                    pixel = struct.unpack_from('>B', file_content, offset=offset)[0]
                    offset += 1
                    image.append(pixel)
            images.append(image)

    labels = []
    with gzip.open(label_filename, 'rb') as f:
        file_content = f.read()
        offset = 0
        magic_num, num = struct.unpack_from('>ii', file_content, offset=offset)
        offset += 8
        assert magic_num == 2049, 'train label error'
        for _ in range(num):
            label = struct.unpack_from('>B', file_content, offset=offset)[0]
            offset += 1
            labels.append(label)
    assert len(images) == len(labels), 'error'

    images = np.array(images, dtype=np.float32)
    images = (images - images.min()) / (images.max() - images.min())
    labels = np.array(labels, dtype=np.uint8)
    return images, labels
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    loss = (-Z[np.arange(0, len(y)), y] + np.log(np.sum(np.exp(Z), axis=1))).sum() / len(y)
    return loss
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_classes = theta.shape[1]
    # import pdb;pdb.set_trace()
    for i in range(int(np.ceil(len(y) / batch))):
        batch_x = X[i*batch: (i+1)*batch]
        batch_y = y[i*batch: (i+1)*batch]
        # 𝑍=normalize(exp(𝑋Θ))
        h = batch_x @ theta
        Z = np.exp(h) / np.sum(np.exp(h), axis=1).reshape(-1, 1)
        d_theta = (np.transpose(batch_x) @ (Z - np.eye(num_classes)[batch_y])) / len(batch_y)
        theta -= lr * d_theta
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    
    num_classes = W2.shape[1]
    # import pdb;pdb.set_trace()
    for i in range(int(np.ceil(len(y) / batch))):
        batch_x = X[i*batch: (i+1)*batch]
        batch_y = y[i*batch: (i+1)*batch]
        Z1 = np.maximum(0, batch_x @ W1)
        h = Z1 @ W2
        G2 = np.exp(h) / np.sum(np.exp(h), axis=1).reshape(-1, 1) - np.eye(num_classes)[batch_y]
        G1 = np.multiply((Z1 > 0), (G2 @ np.transpose(W2)))
        dW1 = np.transpose(batch_x) @ G1 / len(batch_y)
        dW2 = np.transpose(Z1) @ G2 / len(batch_y)
        W1 -= lr * dW1
        W2 -= lr * dW2
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



def train_nn_fn():
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    # print("Training softmax regression")
    # train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    # train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=400, epochs=20, lr=0.2)

def train_nn_fn_cpp():

    import sys
    sys.path.append("src/")
    # Reload the simple_ml module to include the newly-compiled C++ extension
    import importlib
    import simple_ml
    importlib.reload(simple_ml)
    from simple_ml import train_softmax, parse_mnist

    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz", 
                            "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                            "data/t10k-labels-idx1-ubyte.gz")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.2, batch=100, cpp=True)

if __name__ == "__main__":
    train_nn_fn()
    train_nn_fn_cpp()

