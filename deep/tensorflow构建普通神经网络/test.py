import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

def creat_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x, None), name="X")
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name="Y")

    return X, Y

def initialize_parameters():
    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    b1 = parameters['b1']
    b2 = parameters['b2']
    b3 = parameters['b3']



    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3

    return Z3

def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost


def model(X_train, Y_train, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True):
    ops.reset_default_graph()
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    seed = 3
    X, Y = creat_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m/minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatche in minibatches:
                (minibatch_X, minibatch_Y) = minibatche
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))

        parameters = sess.run(parameters)

    return parameters



def main():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    print(X_train_orig.shape)

    index = 0
    plt.imshow(X_train_orig[index])
    print("y = " + str(np.squeeze(Y_train_orig[:, index])))
    # X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_train_flatten = X_train_orig.reshape(-1, X_train_orig.shape[0])
    print(X_train_flatten.shape)
    X_train_flatten = X_train_flatten / 255
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    model(X_train_flatten, Y_train, learning_rate=0.0001, num_epochs=1500, minibatch_size=32, print_cost=True)


if __name__ =="__main__":
    main()



