import tensorflow as tf
import numpy as np


class FNN(object):
    """
    创建一个前馈神经网络
    参数：
    -------------
    learning_rate : float
    drop_out : float
    Layers : list
        the number of layers
    N_hidden :　list
        the number of nodes in layer
    D_input : int
        input dimension
    D_label : int
        label dimension
    Task_type : string
        'regression' or 'classification'
    L2_lambda : float

    """

    def __init__(self, learning_rate, drop_keep, Layers, N_hidden,
                 D_input, D_label, Task_type='regression', L2_lambda=0.0):
        # 全部共有属性
        self.learning_rate = learning_rate
        self.drop_keep = drop_keep
        self.Layers = Layers
        self.N_hidden = N_hidden
        self.D_input = D_input
        self.D_label = D_label
        self.Task_type = Task_type
        self.L2_lambda = L2_lambda
        self.l2_penalty = tf.constant(0.0)
        self.hid_layers = []
        self.W = []
        self.b = []
        self.total_l2 = []
        self.train_step = None
        self.output = None
        self.loss = None
        self.accuracy = None
        self.total_loss = None

        with tf.name_scope('Input'):
            self.inputs = tf.placeholder(tf.float32, [None, D_input], name="inputs")
        with tf.name_scope('Label'):
            self.labels = tf.placeholder(tf.float32, [None, D_label], name='labels')
        with tf.name_scope('keep_rate'):
            self.drop_keep_rate = tf.placeholder(tf.float32, name='dropout_keep')

        self.build('F')

    @staticmethod
    def weight_init(shape):
        """
        Initialize weight of neural network and initialization could be changed here

        :param shape: [in_dim, out_dim]

        :return: a Varible which is initialized by random_uniform
        """
        initial = tf.random_uniform(shape,
                                    minval=-np.sqrt(5) * np.sqrt(1.0 / shape[0]),
                                    maxval=np.sqrt(5) * np.sqrt(1.0 / shape[0]))
        return tf.Variable(initial)

    @staticmethod
    def bias_init(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def variable_summaries(var, name):
        with tf.name_scope(name + '_summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean_' + name, mean)
        with tf.name_scope(name + '_stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('_stddev_' + name, stddev)
        tf.summary.scalar('_max_' + name, tf.reduce_max(var))
        tf.summary.scalar('_min_' + name, tf.reduce_min(var))
        tf.summary.histogram(name=name, values=var)

    def layer(self, in_tensor, in_dim, out_dim, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope(layer_name + '_weights'):
                weights = self.weight_init([in_dim, out_dim])
                self.W.append(weights)
                self.variable_summaries(weights, layer_name + '_weights')

            with tf.name_scope(layer_name + 'biases'):
                biases = self.bias_init([out_dim])
                self.b.append(biases)
                self.variable_summaries(biases, layer_name + '_biases')

            with tf.name_scope(layer_name + '_Wx_plus_b'):
                pre_activate = tf.matmul(in_tensor, weights) + biases
                tf.summary.histogram(layer_name + '_pre_activations', pre_activate)

            activations = act(pre_activate, name='activation')
            tf.summary.histogram(layer_name + '_activations', activations)
            return activations, tf.nn.l2_loss(weights)

    def drop_layer(self, in_tensor):
        dropped = tf.nn.dropout(in_tensor, self.drop_keep_rate)
        return dropped

    def build(self, prefix):
        """
        构建网络
        :param prefix:
        :return:
        """
        incoming = self.inputs
        if self.Layers != 0:
            layer_nodes = [self.D_input] + self.N_hidden
        else:
            layer_nodes = [self.D_input]

        for l in range(self.Layers):
            incoming, l2_loss = self.layer(incoming, layer_nodes[l], layer_nodes[l + 1],
                                           prefix + '_hid_' + str(l + 1), act=tf.nn.relu)
            self.total_l2.append(l2_loss)
            print('Add dense layer: relu with drop_keep:%s' % self.drop_keep)
            print('      %sD --> %sD' % (layer_nodes[l], layer_nodes[l + 1]))
            self.hid_layers.append(incoming)
            incoming = self.drop_layer(incoming)

        if self.Task_type == 'regression':
            out_act = tf.identity
        else:
            out_act = tf.nn.softmax

        self.output, l2_loss = self.layer(incoming, layer_nodes[-1], self.D_label,
                                          layer_name='output', act=out_act)

        print('Add output layer: linear')
        print('   %sD --> %sD' % (layer_nodes[-1], self.D_label))

        with tf.name_scope('total_l2'):
            for l2 in self.total_l2:
                self.l2_penalty += l2
            tf.summary.scalar('l2_penalty', self.l2_penalty)

        if self.Task_type == 'regression':
            with tf.name_scope('SSE'):
                self.loss = tf.reduce_mean((self.output - self.labels) ** 2)
                tf.summary.scalar('loss', self.loss)
        else:
            entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.output,
                                                              labels=self.labels)
            with tf.name_scope('cross_entropy'):
                self.loss = tf.reduce_mean(entropy)
                tf.summary.scalar('loss', self.loss)
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.labels, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)

        with tf.name_scope('total_loss'):
            self.total_loss = self.loss + self.l2_penalty * self.L2_lambda
            tf.summary.scalar('total_loss', self.total_loss)

        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

    @staticmethod
    def shufflelists(lists):
        ri = np.random.permutation(len(lists[1]))
        out = []
        for l in list:
            out.append(l[ri])
        return out


def Standardize(seq):
    """
    :param seq:
    :return:
    """
    # subtract mean
    centerized = seq - np.mean(seq, axis=0)
    # divide standard deviation
    normalized = centerized / np.std(centerized, axis=0)
    return normalized


def Makewindows(indata, window_size=41):
    outdata = []
    mid = int(window_size / 2)
    indata = np.vstack((np.zeros((mid, indata.shape[1])), indata, np.zeros((mid, indata.shape[1]))))
    for index in range(indata.shape[0] - window_size + 1):
        outdata.append(np.hstack(indata[index: index + window_size]))
    return np.array(outdata)


# prepare data for training "XOR"
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 1, 1, 0]
X = np.array(inputs).reshape((4, 1, 2)).astype('int16')
Y = np.array(outputs).reshape((4, 1, 1)).astype('int16')

ff = FNN(learning_rate=1e-3,
         drop_keep=1.0,
         Layers=1,
         N_hidden=[2],
         D_input=2,
         D_label=1,
         Task_type='regression',
         L2_lambda=1e-2)

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log' + '/train', session.graph)

W0 = session.run(ff.W[0])
W1 = session.run(ff.W[1])
print('W_0:\n%s' % session.run(ff.W[0]))
print('W_1:\n%s' % session.run(ff.W[1]))

pY = session.run(ff.output, feed_dict={ff.inputs: X.reshape((4, 2)),
                                       ff.drop_keep_rate: 1.0})
print(pY)

pY = session.run(ff.hid_layers[0], feed_dict={ff.inputs: X.reshape((4, 2)), ff.drop_keep_rate: 1.0})
print(pY)

k = 0.0
for i in range(10000):
    k += 1
    summary, _ = session.run([merged, ff.train_step],
                             feed_dict={ff.inputs: X.reshape((4, 2)),
                                        ff.labels: Y.reshape((4, 1)),
                                        ff.drop_keep_rate: 1.0})
    train_writer.add_summary(summary, k)

W0 = session.run(ff.W[0])
W1 = session.run(ff.W[1])
print('W_0:\n%s' % session.run(ff.W[0]))
print('W_1:\n%s' % session.run(ff.W[1]))

pY = session.run(ff.output, feed_dict={ff.inputs: X.reshape((4, 2)),
                                       ff.drop_keep_rate: 1.0})
print('pY:\n')
print(pY)
