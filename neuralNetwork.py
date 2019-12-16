import numpy
import scipy.special

class NeuralNetwork :
    def __init__(self, nN_settings):
        self.nN_settings = nN_settings

        #初始化输入-隐藏层和隐藏-输出层权重矩阵
        #初始化随机权重采用1/（传入连接数½）作为标准差
        self.wih = numpy.random.normal(0.0, pow(nN_settings.hidden_nodes, -1), (nN_settings.hidden_nodes, nN_settings.input_nodes))
        self.who = numpy.random.normal(0.0, pow(nN_settings.output_nodes, -1), (nN_settings.output_nodes, nN_settings.hidden_nodes))
        # 设置S函数
        self.sfunction = lambda x:scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list, nN_settings):
        self.nN_settings = nN_settings
        # 将输入/目标转为转置向量
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 隐藏层输入计算
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 隐藏层输出计算
        hidden_outputs = self.sfunction(hidden_inputs)

        # 输出层输入计算
        output_inputs = numpy.dot(self.who, hidden_outputs)
        # 隐藏层输出计算
        output_outputs = self.sfunction(output_inputs)

        # 输出层偏差计算
        output_errors = targets-output_outputs
        # 隐藏层偏差计算
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 修正隐藏层到输出层权重
        self.who += nN_settings.learning_rate * numpy.dot((output_errors * output_outputs * (1.0 - output_outputs)),
                                                          numpy.transpose(hidden_outputs))
        # 修正输入层到隐藏层权重
        self.wih += nN_settings.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                                          numpy.transpose(inputs))

    def query(self, inputs_list):
        #将输入转为转置向量
        inputs = numpy.array(inputs_list, ndmin=2).T
        #隐藏层输入计算
        hidden_inputs = numpy.dot(self.wih, inputs)
        #隐藏层输出计算
        hidden_outputs = self.sfunction(hidden_inputs)

        #输出层输入计算
        output_inputs = numpy.dot(self.who, hidden_outputs)
        #隐藏层输出计算
        output_outputs = self.sfunction(output_inputs)

        return output_outputs
