# coding=UTF-8
__author__ = 'wgp'
from keras.datasets import cifar10
from keras.datasets import cifar100
from matplotlib import pyplot
from scipy.misc import toimage
import numpy as np
import math
np.random.seed(1337)  # 好的习惯，设置一个随机数
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
# from keras.utils.visualize_util import plot
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
nb_epoch = 150
nb_class = 10
# pixs of picture
img_rows, img_cols = 32, 32
batch_size = 64
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float64') / 255.0  # 因为加载进来的数据是整型所以我们将它转换成浮点型，并归一化
X_test = X_test.astype('float64') / 255.0
# 将标记转换为二进制矩阵的形式，以更好的解决分类问题
Y_train = np_utils.to_categorical(y_train, nb_class)
Y_test = np_utils.to_categorical(y_test, nb_class)
def My_model():
    """
    定义模型
    :param lr: sgd的学习率
    :param decay: 衰减度
    :param momentum: 冲量
    :return:定义好的模型
    """
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=X_train.shape[1:]))  #1
    model.add(LeakyReLU(0.15))                          #2
    model.add(Convolution2D(32, 3, 3))
    model.add(LeakyReLU(0.15))
    model.add(MaxPooling2D(pool_size=(2, 2)))                #3
    model.add(Dropout(0.25))                                 #4

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(LeakyReLU(0.15))
    model.add(Convolution2D(64, 3, 3))
    model.add(LeakyReLU(0.15))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())                                     #5
    model.add(Dense(512))                                    #6
    model.add(LeakyReLU(0.15))
    model.add(Dropout(0.5))
    model.add(Dense(nb_class))                           
    model.add(Activation('softmax'))  
    print(model.summary())  # 打印出模型的总框
    model_json = model.to_json()
    with open('cifar101.json', 'w') as f:
       f.write(model.to_json())
    sgd = SGD(momentum=0.9, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
def scheduler(epoch):
    """
    自定义函数，通过每次迭代来更新学习率
    :param epoch: 迭代
    :return:
    """
    print(epoch, 'gongpeng')
    initial_rate = 0.02
    drop = 0.5
    epoch_drop = 10.0
    if epoch >= 50:
        _epoch = epoch % 50
        lrate = math.pow(drop, math.floor(_epoch / epoch_drop)) * (initial_rate / math.pow(
            3, math.floor(epoch / 50))) + (0.0085 / math.pow(3, math.floor(epoch / 50)))
    else:
        lrate = math.pow(drop, math.floor(
            (epoch) / epoch_drop)) * initial_rate + 0.0085
    # for i in range(X_train.shape[0]):
    #     alpha = 4.0 / (1.0 + epoch + i) + 0.01
    return lrate
def train_myModel(model):
    """
    训练我们的模型
    :param model: 定义好的模型
    :return:
    """
    lrate = LearningRateScheduler(scheduler)
    modelcheck = ModelCheckpoint(filepath='msgdleakyrelu.hdf5',
                                 monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    earstop = EarlyStopping(
        monitor='val_loss', patience=250, verbose=0, mode='auto')
    # 下面是数据集扩展
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
    )
    datagen.fit(X_train)
    mycifar10_aug = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), samples_per_epoch=X_train.shape[
                                        0], nb_epoch=nb_epoch, validation_data=(X_test, Y_test), callbacks=[earstop, modelcheck,lrate])
    # mycifar10=model.fit(X_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,validation_data=(X_test,Y_test),shuffle=True,callbacks=[earstop,modelcheck])
    model.save_weights('cifar-10.h5', overwrite=True)
    f = open('msgdleakyrelu.txt', 'wb')
    f.write(str(mycifar10_aug.history))
if __name__ == '__main__':
    # showPic()
    model = My_model()
    # model.load_weights('msgdleakyrelu.h5')
    # sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    train_myModel(model)
    # model.load_weights('myCifar10_aug-leakyRelu.h5')
    score = model.evaluate(X_test, Y_test, verbose=1)
    print(score[1])
