# -*- coding: utf-8 -*-

import os
import gc
import sys

import jieba

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
sys.path.append(os.path.realpath(os.path.realpath(__file__)+'/../../../'))
jieba.initialize()


class RandomForest(object):
    """ 随机森林算法实践 """
    def __init__(self, is_save=False):
        self.__is_save = is_save
        self.__clf = None
        self.__train_data_feature = None

        # 加载已保存的训练集
        clf = RandomForestTools.train_data_load()
        train_data_feature = RandomForestTools.feature_data_load()
        if clf and train_data_feature:
            self.__clf = clf
            self.__train_data_feature = train_data_feature

    def build_train_data(self, pre_train_data_list, result_list, train_size=0.9):
        """
        构建训练集
        :param pre_train_data_list: list 需要训练的数据
        :param result_list list 训练数据对应的结果
        :param train_size: float 0<train_size<=1 训练集占总数据的比例
        :return: object 训练集
        """
        # 数据预处理
        print('Start pre-treat data.')
        train_data_list = []
        train_data_feature = set()
        for pre_train_data in pre_train_data_list:
            # 分词
            train_data = self.word_segmentation(pre_train_data)
            train_data_list.append(train_data)
            # 提取分词特征
            for feature in train_data:
                train_data_feature.add(feature)

        # 数据预处理
        data = self.pre_treat_data(train_data_list, train_data_feature)

        # 将训练集随机分成数份，以便自校验训练集准确率
        print('Start split train and test data.')
        data_train, data_test, result_train, result_test = train_test_split(data, result_list, train_size=train_size)

        # 开始训练随机森林,n_jobs设为-1自动按内核数处理数据
        print('Start training random forest.')
        clf = RandomForestClassifier(n_jobs=-1)
        self.__clf = clf.fit(data_train, result_train)
        self.__train_data_feature = train_data_feature
        if self.__is_save:
            # 保存训练集，及各项数据的特征值
            print('Save training result.')
            RandomForestTools.train_data_save(self.__clf)
            RandomForestTools.feature_data_save(train_data_feature)
        print("Build train data finish and accuracy is:%.2f ." %
              (self.__clf.score(data_test, result_test)))

    @staticmethod
    def word_segmentation(train_data):
        """
        分词处理
        :param train_data string 带分词数据
        :return set 分词结果
        """
        word_segmentation_result = set()
        for word in jieba.lcut(train_data):
            word_segmentation_result.add(word)
        return word_segmentation_result

    @staticmethod
    def pre_treat_data(train_data_list, train_data_feature, is_gc_collect=False):
        """
        数据预处理，从已有数据中处理出最终训练集数据
        :param train_data_list: list/set 待训练集内容分词
        :param train_data_feature set 待训练集内容分词特征
        :param is_gc_collect: boolean 数据预处理完成后是否执行垃圾回收
        :return: list 经过预处理的待训练数据
        """
        # 为规避特征保存及取出的过程中顺序打乱而造成数据不对应的情况统一排序
        train_data_feature = sorted(train_data_feature)
        message_list = RandomForestTools.one_hot_encode_feature(train_data_list, train_data_feature)
        if is_gc_collect:
            print('Finish one hot encoder.')
            # 手动执行垃圾回收避免内存占用过高被系统强制kill
            print('Garbage collector: collected %d objects.' % gc.collect())
        return message_list

    def predict(self, predict_data):
        """
        预测输入数据是否为坏样本
        :param predict_data: string 待预测数据
        :return: 预测结果
        """
        predict_data = self.word_segmentation(predict_data)

        data_test = self.pre_treat_data([predict_data], self.__train_data_feature)
        result = self.__clf.predict(data_test)
        return result[0]


class RandomForestTools(object):
    """ 训练集数据操作类 """
    TRAIN_DATA_FILE_DIR = '/tmp/'
    TRAIN_DATA_FILE = 'train_data.pkl'
    FEATURE_DATA_FILE = 'feature_data.pkl'

    @staticmethod
    def train_data_save(clf):
        """
        保存训练集数据
        :param clf:训练集
        :return: boolean True为成功保存，False为保存失败
        """
        filename = RandomForestTools.TRAIN_DATA_FILE_DIR+RandomForestTools.TRAIN_DATA_FILE
        return RandomForestTools.save(filename, clf)

    @staticmethod
    def train_data_load():
        """
        加载已保存的训练集数据
        :return: object/False 训练集数据存在时返回训练集,不存在时返回False
        """
        filename = RandomForestTools.TRAIN_DATA_FILE_DIR+RandomForestTools.TRAIN_DATA_FILE
        if RandomForestTools.exists(filename):
            return RandomForestTools.load(filename)
        else:
            return False

    @staticmethod
    def feature_data_save(feature_data):
        """
        保存特征数据
        :param feature_data: set 训练集的对应的特征数据
        :return: boolean True为成功保存，False为保存失败
        """
        filename = RandomForestTools.TRAIN_DATA_FILE_DIR + RandomForestTools.TRAIN_DATA_FILE
        return RandomForestTools.save(filename, feature_data)

    @staticmethod
    def feature_data_load():
        """
        加载已保存的特征数据
        :return: object 特征数据
        :notice: 加载前应主动检测数据集是否存在，返回的特征值顺序可能会被打乱
        """
        filename = RandomForestTools.TRAIN_DATA_FILE_DIR + RandomForestTools.FEATURE_DATA_FILE
        if RandomForestTools.exists(filename):
            return RandomForestTools.load(filename)
        else:
            return False

    @staticmethod
    def save(filename, python_object):
        if not os.path.isdir(os.path.dirname(filename)):
            os.mkdir(os.path.dirname(filename))
        if joblib.dump(python_object, filename):
            return True
        else:
            return False

    @staticmethod
    def load(filename):
        return joblib.load(filename)

    @staticmethod
    def exists(filename):
        if os.path.isfile(filename):
            return True
        else:
            return False

    @staticmethod
    def one_hot_encode_feature(data_list, data_set):
        """
        根据特征在内容中是否出现将数据格式化成二维二进制数组
        :param data_list: list/set 待格式化数据
        :param data_set: set 特征统计
        :return: list 二维数组
        """
        x, y = 0, 0
        serialize_list = []
        for data in data_list:
            tmp_serialize_list = []
            for key in data_set:
                # 分词以list的方式判断是否存在特征是否存在
                if isinstance(data, list) or isinstance(data, set):
                    tmp_serialize_list.append(1 if key in data else 0)
                elif isinstance(data, basestring) or isinstance(data, int):
                    tmp_serialize_list.append(1 if key == data else 0)
                y += 1
            serialize_list.append(tmp_serialize_list)
            x += 1
        return serialize_list


class RandomForestException(Exception):
    pass

if __name__ == '__main__':
    pre_train_data = [
        u'我很开心',
        u'我非常开心',
        u'我其实很开心',
        u'我特别开心',
        u'我超级开心',
        u'我不开心',
        u'我一点都不开心',
        u'我很不开心',
        u'我非常不开心',
        u'我很久没那么不开心了',
    ]
    result_list = [
        u'开心',
        u'开心',
        u'开心',
        u'开心',
        u'开心',
        u'不开心',
        u'不开心',
        u'不开心',
        u'不开心',
        u'不开心',
    ]
    rf = RandomForest()
    rf.build_train_data(pre_train_data,result_list)
    print(rf.predict(
        u'你猜我开心吗?',
    ))
