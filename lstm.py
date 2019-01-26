from __future__ import print_function

import tensorflow as tf
import scipy.io as sio
import numpy as np

X4 = sio.loadmat('C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/video/Images/Final/ts.mat')
X4=X4['Data']
import csv
with open('C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/video/Images/Final/lab.csv', 'r') as mf:

     re = csv.reader(mf,delimiter=',',quotechar='|')
     re=np.array(list(re))
     label = re.astype(np.float64)
     label=np.squeeze(label)  
     
y_train=label[:158]
y_test=label[158:]

data=X4
data = data.astype(np.float64)

data_d=np.zeros((206, 100))

import lttb
for i in range(206):
    Otr=(data[i,:])
    Otr2=Otr.shape[0]
    Otr=np.array([range(Otr2), Otr.T]).T
    OOtr= lttb.downsample(Otr, n_out=100)
    Qtr=OOtr[:,1].T
    data_d[i,:]=np.squeeze(Qtr)-1

#x_train=data_d[:158]
#x_test=data_d[158:]

from sklearn.model_selection import KFold
m=0

kf=KFold(3, random_state=True, shuffle=False)
kf.get_n_splits(data_d)

k=0
for train_index, test_index in kf.split(data_d):
    x_train, x_test = data_d[train_index], data_d[test_index]
    y_train, y_test = label[train_index], label[test_index] 
    if k==m:
       break 
    k=k+1
    
#seed = 7
#np.random.seed(seed)

#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#scaler = MinMaxScaler(feature_range=(0, 1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

x_train=x_train.reshape((x_train.shape[0], 100, 1))
x_test=x_test.reshape((x_test.shape[0], 100, 1))
tf.reset_default_graph() 
class ToySequenceData(object):        
    def __init__(self, XX, l2, n_samples, max_seq_len=100):
        
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            # Random sequence length
            lenn = 100
            self.seqlen.append(lenn)

            self.data.append(XX[i,:])

            self.batch_id = 0
            if(l2[i]==1):
              self.labels.append([1., 0.])
            else:
              self.labels.append([0., 1.])

    def next(self, batch_size):

        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen

# Parameters
learning_rate = 0.005
training_iters = 50000
batch_size = 8
display_step = 20

# Network Parameters
seq_max_len = 100 # Sequence max length
n_hidden = 8 # hidden layer num of features
n_classes = 2 # linear sequence or not

trainset = ToySequenceData(x_train, y_train, n_samples=x_train.shape[0], max_seq_len=seq_max_len)
testset = ToySequenceData(x_test, y_test, n_samples=x_test.shape[0], max_seq_len=seq_max_len)

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def dynamicRNN(x, seqlen, weights, biases):

    # data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_3 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2,  lstm_cell_3], state_is_tuple=True)
    
    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, x, dtype=tf.float32,
                                sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    
    outputs =  tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

pred = dynamicRNN(x, seqlen, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#GradientDescent
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
                                                seqlen: batch_seqlen})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
                                             seqlen: batch_seqlen})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                   seqlen: test_seqlen   }))



