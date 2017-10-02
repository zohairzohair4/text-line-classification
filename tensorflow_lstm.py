import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,StandardScaler


learning_rate = 0.001
display_step = 2

num_input = 1
num_hidden = 200 # hidden layer num of features
num_classes = 12 



dict_y = {
0: 'alice_in_wonderland',
1: 'dracula',
2: 'dubliners',
3: 'great_expectations',
4: 'hard_times',
5: 'huckleberry_finn',
6: 'les_miserable',
7: 'moby_dick',
8: 'oliver_twist',
9: 'peter_pan',
10:' talw_of_two_cities',
11:' tom_sawyer',
}


def convert_chars(path):
    data_list = []
    dataX = [] 
    data_final_X = []
    flat_list = []

    f = open(path)
    for data in f.readlines():
            data_list.append(list(data.strip('\n')))
    f.close()
    
    
    
            
    for sublist in data_list:
        for item in sublist:
            flat_list.append(item)        
            
            
            
    list_of_characters = set(flat_list)   
    
    
    
    # create mapping of characters to integers (0-25) and the reverse
    char_to_int = dict((c, i) for i, c in enumerate(list_of_characters))
    int_to_char = dict((i, c) for i, c in enumerate(list_of_characters))   
    
       
    for data_final in data_list:
        for x in data_final:
            dataX.append(char_to_int[x])
        data_final_X.append(dataX)
        dataX = []
    
    return data_final_X    
    
def zero_padding(input_array,max_length):
    result = np.zeros([len(input_array), max_length], dtype=np.int32)
    max_len_array = np.zeros([len(input_array)], dtype=np.int32)
    for i, x_i in enumerate(result):
            result[i,:len(input_array[i])] = input_array[i]
            max_len_array[i] = len(input_array[i])   
    return result, max_len_array      
    


x_train = convert_chars("xtrain_obfuscated.txt") 
x_test = convert_chars("xtest_obfuscated.txt")


seq_max_len = max(len(max(x_train,key=len)), len(max(x_test,key=len)))
x_train_padded,seq_len_array = zero_padding(x_train,seq_max_len)
x_test,seq_len_array_test = zero_padding(x_test,seq_max_len)


    
data  = tf.placeholder('float', [None, seq_max_len, num_input])
target = tf.placeholder('float', [None, num_classes])
seqlen = tf.placeholder(tf.int32, [None])


weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


y_train = []
f = open('ytrain.txt')
for temp in f.readlines():
        y_train.append(temp.strip('\n'))
f.close()
y_train = list(map(int, y_train))

lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_train)

#x_train_padded = x_train_padded[0:1000]
#y_train = y_train[0:1000]
#seq_len_array = seq_len_array[0:1000]

scaler = StandardScaler()
x_train_padded = scaler.fit_transform(x_train_padded)

train_data, test_data, train_label, test_label = train_test_split(
            x_train_padded, y_train, test_size=0.33, random_state=42)

seq_len_array_train, seq_len_array_test, _, _ = train_test_split(seq_len_array,seq_len_array, test_size=0.33, random_state=42)


    


def LSTM(x, seqlen, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    
    x = tf.unstack(x, seq_max_len, 1)
    #lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    
    
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    #lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)
    '''
    stacked_rnn = []
    for layer in range(3):
        stacked_rnn.append(rnn.BasicRNNCell(num_hidden))
    rnn_cells= tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
    '''
    

    # Get lstm cell output
    #outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,sequence_length=seqlen)    
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32,sequence_length=seqlen)
    
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
    outputs = tf.gather(tf.reshape(outputs, [-1, num_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']
    '''
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
    '''


logits = LSTM(data,seqlen, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=target))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()




with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    
    
    batch_size = 256
    no_of_batches = int(len(train_data)/batch_size)
    epoch = 100
    train_data = train_data.reshape((train_data.shape[0], seq_max_len, num_input))
    for step in range(epoch):
        ptr = 0
        for j in range(no_of_batches):
            inp,seq, out = train_data[ptr:ptr+batch_size], seq_len_array_train[ptr:ptr+batch_size], train_label[ptr:ptr+batch_size]
            
            sess.run(train_op,{data: inp, seqlen:seq, target: out})
            
            ptr+=batch_size
            
            
        
        # Run optimization op (backprop)
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={data: inp,
                                                                 seqlen:seq,
                                                                 target: out})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
    test_data = test_data.reshape((test_data.shape[0], seq_max_len, num_input))
    # Calculate accuracy for 128 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={data: test_data, seqlen:seq_len_array_test, target: test_label}))






















