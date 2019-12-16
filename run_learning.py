from random import randint

import matplotlib.pyplot
import numpy
from settings import Settings
from neuralNetwork import NeuralNetwork

def run_learning():
    nN_settings = Settings()

    nLearning = NeuralNetwork(nN_settings)
    #导入训练集
    training_data_file = open("mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # 开始训练
    # 训练5次
    epochs = nN_settings.training_times
    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(nN_settings.output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            nLearning.train(inputs, targets, nN_settings)

    #导入测试集
    test_data_file = open("mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    #测试网络
    scorecard = []
    i = 0
    j = randint(0, 9900)
    for record in test_data_list:
        all_values = record.split(',')
        correst_label = int(all_values[0])
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = nLearning.query(inputs)
        label = numpy.argmax(outputs)
        # 抽取10个图样检查效果
        i += 1
        if i > j and i < (j+3):
            image_array =numpy.asfarray(all_values[1:]).reshape((28,28))
            title = 'system:' + str(label) + ', answer: ' + str(correst_label)
            matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
            matplotlib.pyplot.title(title, fontsize = 20)
            matplotlib.pyplot.show()
        if label == correst_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    scorecard_array = numpy.asarray(scorecard)
    print('performance = ', scorecard_array.sum() / scorecard_array.size)
    savechoose = input("save or not [Y/N]:")
    if savechoose == 'Y' or savechoose == 'y':
        fwih = input("请输入输入层-隐藏层权重文件名：")
        filewih = str(fwih) + '.npy'
        fwho = input("请输入隐藏层-输出层权重文件名：")
        filewho = str(fwho) + '.npy'
        numpy.save(filewih, nLearning.wih)
        numpy.save(filewho, nLearning.who)

    else:
        print('Exit')


def run_demo():
    demo_settings = Settings()
    demo_learning = NeuralNetwork(demo_settings)
    demo_wih_list = numpy.load('wih.npy')
    demo_who_list = numpy.load('who.npy')
    demo_learning.who = demo_who_list
    demo_learning.wih = demo_wih_list

    #导入测试集
    test_data_file = open("mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    #测试网络
    scorecard = []
    i = 10
    j = randint(0, 9900)
    for record in test_data_list:
        all_values = record.split(',')
        correst_label = int(all_values[0])
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = demo_learning.query(inputs)
        label = numpy.argmax(outputs)
        # 抽取10个图样检查效果
        i += 1
        if i > j and i < (j+5):
            image_array =numpy.asfarray(all_values[1:]).reshape((28,28))
            title = 'system:' + str(label) + ', answer: ' + str(correst_label)
            matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
            matplotlib.pyplot.title(title, fontsize = 20)
            matplotlib.pyplot.show()
        if label == correst_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    scorecard_array = numpy.asarray(scorecard)
    print('performance = ', scorecard_array.sum() / scorecard_array.size)
