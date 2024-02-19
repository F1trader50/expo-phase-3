import numpy as np
import pandas as pd

# Load and preprocess the data
data = pd.read_csv(r'C:\Users\rohan\PycharmProjects\Expo-Phase-3\train.csv')
data = np.array(data)
np.random.shuffle(data)
X_dev = data[:1000, 1:].T / 255
Y_dev = data[:1000, 0].T
X_train = data[1000:, 1:].T / 255
Y_train = data[1000:, 0].T



def init_params(width, layers):
    W = [np.zeros((width, width)).tolist()] * layers
    b = [np.zeros((width, 1)).tolist()] * layers
    A = [np.zeros(784) if i == 0 else np.zeros(10) if i == layers + 1 else np.zeros(width) for i in range(layers+2)]
    z = [np.zeros(784) if i == 0 else np.zeros(10) if i == layers + 1 else np.zeros(width) for i in range(layers+2)]
    for i in range(layers):
        if i == 0:
            W[i] = np.random.rand(width, 784) * 2 - 1
            b[i] = np.random.rand(1, 784) * 2 - 1
        elif i == layers - 1:
            W[i] = np.random.rand(width, width) * 2 - 1
            b[i] = np.random.rand(1, width) * 2 - 1
        else:
            W[i] = np.random.rand(10, width) * 2 - 1
            b[i] = np.random.rand(1, 10) * 2 - 1


    return W, b, A, z


def ReLU(z):
    return np.maximum(z, 0)


def softmax(z):
    A = np.exp(z) / sum(np.exp(z))
    return A


def forward_prop(W, b, X, width, layers, A, z):

    for i in range(layers):
        if i == 0:
            #W has a shape of (width, 784) 1, (Width, width), (10, width)
            #A has shape (Width, layers + 1)
            print(X[376][80])

            z[i] = W[i].dot(A[i]) + b[i]
            A[i + 1] = ReLU(z[i])
        elif i == layers - 1:
            z[i] = W[i].dot(A[i]) + b[i]
            A[i + 1] = softmax(z[i])
        else:
            z[i] = W[i].dot(A[i]) + b[i]
            A[i + 1] = ReLU(z[i])
    return z, A


def ReLU_deriv(z):
    return z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(z, A, W, X, Y, width, layers):
    one_hot_Y = one_hot(Y)
    dz = [np.zeros_like(z[i]) for i in range(layers)]
    db = [np.zeros((width, 1)) for i in range(layers)]
    dW = [np.zeros_like(W[i]) for i in range(layers)]

    for i in range(layers - 1, -1, -1):
        if i == layers - 1:
            dz[i] = A[i + 1] - one_hot_Y
            dW[i] = 1 / m_train * dz[i].dot(A[i].T)
            db[i] = 1 / m_train * np.sum(dz[i], axis=1, keepdims=True)
        else:
            dz[i] = W[i + 1].T.dot(dz[i + 1]) * ReLU_deriv(z[i])
            dW[i] = 1 / m_train * dz[i].dot(A[i].T)
            db[i] = 1 / m_train * np.sum(dz[i], axis=1, keepdims=True)

    return dW, db


def update_params(W, b, dW, db, alpha, width, layers):
    for i in range(layers):
        W[i] = W[i] - alpha * dW[i]
        b[i] = b[i] - alpha * db[i]

    return W, b


def get_predictions(A):
    return np.argmax(A[-1], 0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def outliers(weights, threshold):
    for layer in weights:
        for neuron in layer:
            if sum(abs(neuron)) < threshold:
                del neuron
                return len(weights)
    #return True


# def cut_off(A):
#     B = []
#
#     for layer in A:
#         for activation in layer:
#             B.append(activation)
#
#     average = np.mean(B)
#     std_dev = np.std(B)
#     outliers = search_outliers(B, 2)
#     print("Outliers: ", outliers)
#     print("Average:", average)
#     print("Standard Deviation:", std_dev)


# def grow():
#     return

def gradient_descent(X, Y, alpha, iterations, width, layers):
    W, b, A, z = init_params(width, layers)
    for i in range(iterations):
        z, A = forward_prop(W, b, X, width, layers, A, z)
        dW, db = backward_prop(z, A, W, X, Y, width, layers)
        W, b = update_params(W, b, dW, db, alpha, width, layers)
        print("Iteration: ", i)
        predictions = get_predictions(A)
        print('Accuracy: ', get_accuracy(predictions, Y))
        print('W: ', len(W[0][1]))
        outlier = outliers(W, 5)
        print('Outliers: ', outlier)
    return W, b


W, b = gradient_descent(X_train, Y_train, 1, 3, 20, 3)
