import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MLP(object):
    def __init__(self):
        print("""
                    ███╗   ███╗██╗     ██████╗ 
                    ████╗ ████║██║     ██╔══██╗
                    ██╔████╔██║██║     ██████╔╝
                    ██║╚██╔╝██║██║     ██╔═══╝ 
                    ██║ ╚═╝ ██║███████╗██║     
                    ╚═╝     ╚═╝╚══════╝╚═╝   
        """)
        # 网络参数
        self.learning_rate = 0.001  # 学习率
        self.max_iter = 10000  # 最大迭代次数
        self.n_hidden_1 = 2  # 第一层神经元个数
        self.n_input = 2  # 样本特征数
        # 定义权值和偏置
        self.Weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1]), name='layer1_w'),
            'out': tf.Variable(tf.random_normal([self.n_hidden_1, 1]), dtype=tf.float32)
        }
        self.biases = {
            'h1': tf.Variable(tf.zeros([1, self.n_hidden_1]), name='layer1_bias'),
            'out': tf.constant(0.)
        }
        self.model_path = "./model/model.ckpt"  # 模型保存路径
        self.names = ['h1', 'out']  # 便与遍历
        return

    def __add_layer__(self, name, inputs, activation_function=None):
        """
        添加一个神经网络层
        :param inputs: 输入数据
        :param activation_function: 激活函数
        :return: 该层输出
        """
        ys = tf.matmul(inputs, self.Weights[name]) + self.biases[name]
        if activation_function is None:
            outputs = ys
        else:
            outputs = activation_function(ys)
        self.outputs = outputs
        return self.outputs

    def fit(self, X_train, y_train):
        """
        训练分类器
        :param X_train:训练样本
        :param y_train:训练标签
        :return:
        """
        X = tf.placeholder(tf.float32, [None, 2], name='X_train')
        y = tf.placeholder(tf.float32, [None, 1], name='y_train')
        # init_weight = tf.assign(self.Weights['h1'], np.array([[1., 10.], [-10., 1.]]))
        # 使用sigmoid代替hardlimit
        layer1 = self.__add_layer__('h1', X, activation_function=tf.nn.sigmoid)
        # 输出层
        predict = self.__add_layer__('out', layer1)
        # 定义损失函数
        loss = tf.reduce_mean(tf.sigmoid(tf.abs(predict - y)))
        # 定义优化器
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
        # 定义保存器
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            # 可视化Loss
            tf.summary.scalar('Loss', loss)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("tensorboard/", sess.graph)
            # sess.run(init_weight)
            # 开始训练
            for epoch in range(self.max_iter):
                X_train = X_train.reshape((-1, 2))
                y_train = y_train.reshape((-1, 1))
                sess.run(optimizer, feed_dict={X: X_train, y: y_train})
                if (epoch + 1) % 500 == 0:
                    l = sess.run(loss, feed_dict={X: X_train, y: y_train})
                    r = sess.run(merged, feed_dict={X: X_train, y: y_train})
                    print("Epoch:", '%05d' % (epoch + 1), "loss=", "{:.3f}".format(l))
                    writer.add_summary(r, epoch)
            writer.close()
            print("Optimization Finished!")
            training_loss = sess.run(loss, feed_dict={X: X_train, y: y_train})
            print("Training loss=", training_loss, '\n')
            res = np.around(sess.run(predict, feed_dict={X: X_train})).reshape((1, -1))
            print("Training result: ", res)
            # saver.save(sess, self.model_path)
            print("Model saved at: ", self.model_path)
            print("""  
            [Input]                                                 
                |-Weight({0})-bias:({1})
                |-Weight({2})-bias:({3})
                |
                V
            [Layer1]
                |-Weight({4})-bias:({5})
                |
                V
            [Output] 
            """.format(self.Weights['h1'].eval()[:, 0].reshape((1, -1)),
                       self.biases['h1'].eval()[0, 0],
                       self.Weights['h1'].eval()[:, 1].reshape((1, -1)),
                       self.biases['h1'].eval()[0, 1],
                       self.Weights['out'].eval().reshape((1, -1)),
                       self.biases['out'].eval()))
        return

    def get_params(self):
        """
        输出网络参数
        :return: 权值，偏置
        """
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_path)  # 恢复模型
            weight = self.Weights.copy()
            bias = self.biases.copy()
            for name in self.names:
                weight[name] = self.Weights[name].eval()
                bias[name] = self.biases[name].eval()
        return weight, bias

    def predict(self, X_test):
        """
        使用模型预测
        :param X_test: 测试数据
        :return: 预测结果
        """
        # 重建网络
        X = tf.placeholder(tf.float32, [None, 2])
        layer1 = self.__add_layer__('h1', X, activation_function=tf.nn.sigmoid)
        # 输出层
        predict = self.__add_layer__('out', layer1)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_path)  # 恢复模型
            print("""
            [Input]                                                 
                |-Weight({0})-bias:({1})
                |-Weight({2})-bias:({3})
                |
                V
            [Layer1]
                |-Weight({4})-bias:({5})
                |
                V
            [Output] 
            """.format(self.Weights['h1'].eval()[:, 0].reshape((1, -1)),
                       self.biases['h1'].eval()[0, 0],
                       self.Weights['h1'].eval()[:, 1].reshape((1, -1)),
                       self.biases['h1'].eval()[0, 1],
                       self.Weights['out'].eval().reshape((1, -1)),
                       self.biases['out'].eval()))
            result = np.around(np.abs(sess.run(predict, feed_dict={X: X_test}))).reshape((1, -1))  # 预测
            print("[Result]: ", result)
        return result


def hardlim(x):
    """
    硬极限函数
    :param x: 数组
    :return: 数组
    """
    x = np.where(x < 0, x, 1)
    x = np.where(x > 0, x, 0)
    return x


def draw(weight, bias):
    """
    可视化结果
    :param weight: 网络权值矩阵
    :param bias: 网络偏置矩阵
    :return:
    """
    weight = weight['h1']
    bias = bias['h1']
    fig = plt.figure()
    ax = Axes3D(fig)
    xs = np.arange(-2.5, 2.5, 0.1)
    ys = np.arange(-2.5, 2.5, 0.1)
    xs, ys = np.meshgrid(xs, ys)
    zs1 = hardlim(weight[0, 0] * xs + weight[1, 0] * ys + bias[0, 0])
    zs2 = hardlim(weight[0, 1] * xs + weight[1, 1] * ys + bias[0, 1])
    zs = 2 * zs1 + zs2
    ax.plot_surface(xs, ys, zs, cmap='rainbow')
    plt.show()
    return


if __name__ == '__main__':
    X = np.array([[1, 1], [1, 2], [2, -1], [2, 0], [-1, 2], [-2, 1], [-1, -1], [-2, -2]])
    y = np.array([[0], [0], [3], [3], [1], [1], [2], [2]])
    # mlp = MLP()
    # mlp.max_iter = 30000
    # mlp.learning_rate = 0.01
    # mlp.fit(X, y)
    # result = mlp.predict(X)
    # weight, bias = mlp.get_params()
    # draw(weight, bias)

    color = y.copy().astype(np.str)
    color[np.where(y == 0)[0]] = 'r'
    color[np.where(y == 1)[0]] = 'c'
    color[np.where(y == 2)[0]] = 'm'
    color[np.where(y == 3)[0]] = 'b'
    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(color), marker='.')
    plt.show()
