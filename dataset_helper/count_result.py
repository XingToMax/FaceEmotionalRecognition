from labels import *

class TestResult:
    def __init__(self):
        # 测试集各表情数量
        self.sample_num_count = TestResult.init_list()
        # 各表情正确数量
        self.real_num_count = TestResult.init_list()
        # 识别总数
        self.sum = 0
        # 正确总数
        self.correct = 0
        # 各表情识别情况
        self.recognition_list = [TestResult.init_list() for x in range(emote_kind_num)]

    def update(self, label_test, label_res):
        for i in range(len(label_test)):
            self.sample_num_count[label_test[i]] = self.sample_num_count[label_test[i]] + 1
            self.recognition_list[label_test[i]][label_res[i]] = self.recognition_list[label_test[i]][label_res[i]] + 1
            self.sum = self.sum + 1
            if label_test[i] == label_res[i]:
                self.real_num_count[label_test[i]] = self.real_num_count[label_test[i]] + 1
                self.correct = self.correct + 1

    def display(self):
        for i in range(emote_kind_num):
            print(emote_labels[i], '总数 :', self.sample_num_count[i], '正确率 :',  TestResult.modify_to_percent(self.real_num_count[i] / self.sample_num_count[i]))
        table_names = '\t'
        for i in range(emote_kind_num):
            table_names += emote_labels[i]
            table_names += ' '
        print(table_names)
        for i in range(emote_kind_num):
            col_val = ''
            for j in range(emote_kind_num):
                col_val += TestResult.modify_to_percent(self.recognition_list[i][j] / self.sample_num_count[i])
                col_val += ' '
            print(emote_labels[i], col_val)

    @staticmethod
    def init_list():
        list = []
        for i in range(emote_kind_num):
            list.append(0)
        return list

    @staticmethod
    def modify_to_percent(val):
        return format(val, '.2%')