class Settings:
    def __init__(self):
        '''初始化神经网络每层节点数和学习速率'''
        self.input_nodes = 784
        self.hidden_nodes = 100
        self.output_nodes = 10
        self.learning_rate = 0.1
        self.training_times = 3