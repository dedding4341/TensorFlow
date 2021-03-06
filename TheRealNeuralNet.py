import tensorflow as tf
import collect
import numpy as np

#from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

'''
10 classes, 0-9
'''


#Setting up neural network
n_nodes_hl1 = 100
n_nodes_hl2 = 150
n_nodes_hl3 = 125

n_classes = 2
batch_size = 80 #setup batches, batches of 100 images, up this to compare more at one time
data_num = 8000

#height x width
x = tf.placeholder(tf.float32, [batch_size, 500]) # don't know what the fixed width is
y = tf.placeholder(tf.float32, [batch_size,2]) # temporary values

def neural_network_model(data):
    #input_data * weights + biases

    #Biases exist to make the neural network a bit more dynamic
    hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([500, n_nodes_hl1])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases' : tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) , hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) , hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) , hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction))


    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #Cycles feed forward +  back prop
    hm_epochs = 2
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(data_num/batch_size)):
                epoch_x, epoch_y = collect.nextbatch(batch_size) #Lol it aint this easy :)

                print('New batch beginning!')

                #print(np.shape(x), np.shape(y))
                #print(len(np.array(epoch_x)))
                #print(len(np.array(epoch_y)))
                _, c, predictnumpy = sess.run([optimizer, cost, prediction], feed_dict = {x: np.array(epoch_x), y: np.array(epoch_y)})
                epoch_loss += c
                saver.save(sess, 'checkpoints/the-best.ckpt')
            print('Epoch ', epoch, ' completed out of ', hm_epochs, ' loss: ', epoch_loss)
            correct = np.equal(np.argmax(predictnumpy,0), np.argmax(y,0))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            print('Accuracy: ', sess.run(accuracy.eval()))


train_neural_network(x)
