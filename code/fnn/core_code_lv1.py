import tensorflow as tf
import numpy as np

#构建网络
#输入维度
D_input = 2
#输出
D_label = 1
#隐藏层神经元数
D_hidden = 2
#学习率
lr = 1e-4

x = tf.placeholder(dtype=tf.float32, shape=[None, D_input], name='input')
label = tf.placeholder(dtype=tf.float32, shape=[None, D_label], name='label')

#tensorflow中的变量tf.Variable是用于定义在训练过程中可以更新的值
W_h1 = tf.Variable(tf.truncated_normal(shape=[D_input, D_hidden], stddev=0.1), name='W_h')
b_h1 = tf.Variable(tf.constant(0.1, tf.float32, shape=[D_hidden]), name='b_h')
pre_act_h1 = tf.matmul(x, W_h1) + b_h1
act_h1 = tf.nn.relu(pre_act_h1, "act_h1")

W_o = tf.Variable(tf.truncated_normal(shape=[D_hidden, D_label], stddev=0.1), name='W_o')
b_o = tf.Variable(tf.constant(0.1, tf.float32, shape=[D_label], name='b_o'))
pre_act_o = tf.matmul(act_h1, W_o) + b_o
y = tf.nn.relu(pre_act_o, name='act_y')

loss = tf.reduce_mean((y - label) ** 2)

train_step = tf.train.AdamOptimizer(lr).minimize(loss)

#输入数据
X = [[0, 0],[0, 1],[1, 0],[1, 1]]
Y = [[0], [1], [1], [0]]

X = np.array(X).astype(dtype='int16')
Y = np.array(Y).astype(dtype='int16')

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

'''
https://www.cnblogs.com/fonttian/p/7342388.html
其改变是在第一遍的时候,仔细看源代码,输出的Train_Loss与
后面sess.run([Train_Step,Train_Loss], feed_dict={X: batch_xs, Y_true: batch_ys})中的’Train_Loss’同名,
显然第一遍运行之后,原本应该传入session的Train_Loss就从张量变成了 float32
'''

'''
GD: X和Y是4组不同的训练数据。上面将所有数据输入到网络，算出平均梯度来更新一次网络的方法叫做GD。
缺点：效率很低，也容易卡在局部极小值，
优点：但更新方向稳定。
'''
for i in range(100000):
    loss_value, _ = sess.run([loss, train_step], feed_dict={x:X, label:Y})
    print('step', i ,': ' , loss_value)

'''
SGD（Gradient Descent）：一次只输入一个训练数据到网络，算出梯度来更新一次网络的方法叫做SGD。
优点：效率高，适合大规模学习任务，容易挣脱局部极小值（或鞍点）
缺点：但更新方向不稳定
'''
epoch = 100000
for i in range(epoch):
    for j in range(X.shape[0]):
        loss_value, _ = sess.run([loss, train_step], feed_dict={x: X[j], label: Y[j]})
        print('step', i, ': ', loss_value)

'''
batch-GD：这是上面两个方法的折中方式。每次计算部分数据的平均梯度来更新权重。
部分数据的数量大小叫做batch_size，对训练效果有影响。一般10个以下的也叫mini-batch-GD。
'''
epoch = 10000  #训练几个epoch
batch_index = 0
batch_size = 2
for i in range(epoch):
    while batch_index <= X.shape[0]:
        loss_value, _ = sess.run([loss, train_step],
                                 feed_dict={x: X[batch_index:batch_index + batch_size],
                                            label: Y[batch_index: batch_index + batch_size]})
        batch_index += batch_size

'''
shuffle：SGD和batch-GD由于只用到了部分数据。若数据都以相同顺序进入网络会使得随后的epoch影响很小。
shuffle是用于打乱数据在矩阵中的排列顺序，提高后续epoch的训练效果。
'''
#shuffle函数
def shuffle(data_list):
    index = np.random.permutation(len(data_list[1]))
    ret = []
    for data in data_list:
        ret.append(data[index])
    return ret

epoch = 10000
batch_index = 0
batch_size = 2
for i in range(epoch):
    X, Y = shuffle([X, Y])
    while batch_index <= len(X):
        loss_value, _ = sess.run([loss, train_step], feed_dict={x: X[batch_index, batch_index+batch_size],
                                                                label: Y[batch_index, batch_index + batch_size]})
        batch_index += batch_size

predict = sess.run(y, feed_dict={x: X, label: Y})
print('predict: ', predict)





















