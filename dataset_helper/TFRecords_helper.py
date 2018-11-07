import cv2
import numpy as np
import tensorflow as tf
import sys

from labels import *
from dataset_helper.IO_helper import *

face_patterns = cv2.CascadeClassifier(
        os.getcwd() + '/res/haarcascade_frontalface_default.xml')

# recorder 的文件路径
root_recorder_path = '../datasets/'
train_recorder_path = root_recorder_path + 'train.tfrecords'
val_recorder_path = root_recorder_path + 'val_tfrecords'
test_recorder_path = root_recorder_path + 'test_tfrecords'
train_recorder_enhance_path = root_recorder_path + 'train_enhance.tfrecords'
train_recorder_enhance_with_test_path = root_recorder_path + 'train.enhance_with_test.tfrecords'


# 将数据转化成对应的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# 创建训练集
def create_train_tf_records():
    resource = read_train_images()
    resource.shuffle_data()
    create_tf_records(resource, train_recorder_path)
    print("train num : ", len(resource.un_seq_data))

def create_train_tf_records_enhance():
    resource = read_enhance_train_images()
    resource.shuffle_data()
    create_tf_records(resource, train_recorder_enhance_path)
    print("train num : ", len(resource.un_seq_data))

def create_train_tf_records_enhance_with_test():
    resource = read_enhace_train_with_test_images()
    resource.shuffle_data()
    create_tf_records(resource, train_recorder_enhance_with_test_path)
    print("train num : ", len(resource.un_seq_data))

# 创建测试集
def create_test_tf_records():
    resource = read_test_images()
    resource.shuffle_data()
    print("test num : ", len(resource.un_seq_data))
    create_tf_records(resource, test_recorder_path)


# 创建验证集
def create_val_tf_records():
    resource = read_val_images()
    resource.shuffle_data()
    print("val num : ", len(resource.un_seq_data))
    create_tf_records(resource, val_recorder_path)


# 依赖ImageDataResource创建tfrecords，输出到recorder_path
def create_tf_records(resource, recorder_path):
    writer = tf.python_io.TFRecordWriter(recorder_path)
    for i in range(len(resource.un_seq_data)):
        if not i % 1000:
            print('data: {}/{}'.format(i, len(resource.un_seq_data)))
            sys.stdout.flush()
        img = resource.un_seq_data[i]
        img.encode_image()
        # 创建一个属性
        feature = {'label': _int64_feature(img.label),
                   'image': _bytes_feature(tf.compat.as_bytes(img.data.tostring()))}

        # 创建一个 example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # 将上面的example protocol buffer写入文件
        writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()


# 从tf recordes 获取数据
def decode_from_tf_records(filename_queue, is_batch, batch_size = 50, shape = image_shape_):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image' : tf.FixedLenFeature([], tf.string),
                                       })  #取出包含image和label的feature对象
    image = tf.decode_raw(features['image'],tf.float32)
    image = tf.reshape(image, shape)
    label = tf.cast(features['label'], tf.int32)

    if is_batch:
        min_after_dequeue = 50
        capacity = min_after_dequeue + 3 * batch_size
        image, label = tf.train.shuffle_batch([image, label],
                                                          batch_size=batch_size,
                                                          num_threads=3,
                                                          capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
    return image, label


# 对数据集做一个简单处理,强耦合
def create_one_hot(num, sum):
    label = []
    for i in range(sum):
        if i == num:
            label.append(1)
        else:
            label.append(0)
    return label

# 对数据集做一个简单处理,强耦合
def modify_size(input_image, input_label, shape = [1764]):
    images = []
    labels = []
    for img in input_image:
        images.append(np.reshape(img, shape))
    for label in input_label:
        labels.append(create_one_hot(label, 7))
    return images, labels


# 更新每轮的测试结果
# pre : 预测结果
# label : 标签
#  acc_record : 各种表情正确的数量
#  record : 各种表情的总数
def check_accuracy(pre, label, acc_record, record):
    for i in range(len(pre)):
        if pre[i] == label[i] :
            acc_record[label[i]] = acc_record[label[i]] + 1
        record[label[i]] = record[label[i]] + 1

    return acc_record, record

# 人脸探测
def face_detect(image):
    faces = face_patterns.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    return faces

# 人脸表情标定
def mark_human_emote(img, coors, emotes):
    x_limit, y_limit = img.shape[0 : 2]
    for i in range(len(coors)):
        x ,y, w, h = coors[i]
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
        text_beg_x = x
        text_beg_y = y
        img = cv2.putText(img, emotes[i], (text_beg_x, text_beg_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    return img

if __name__ == '__main__':
    # create_train_tf_records()
    # create_test_tf_records()
    # create_val_tf_records()
    # make train.tfrecord
    create_train_tf_records_enhance_with_test()
    # image = cv2.imread('F:/ToMax/study/AI/datasets/extended-cohn-kanade-images/image/S005/001/S005_001_00000010.png', 1)
    # faces = face_detect(image)
    # image = mark_human_emote(image, faces, ['happy'])
    # cv2.imshow('face', image)
    # cv2.resizeWindow('face', 800, 600)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        # filename_queue = tf.train.string_input_producer([test_recorder_path], num_epochs=None)  # 读入流中
    # train_image, train_label = decode_from_tf_records(filename_queue, is_batch=True)
    # with tf.Session() as sess:  # 开始一个会话
    #     init_op = tf.global_variables_initializer()
    #     sess.run(init_op)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #
    #     try:
    #         # while not coord.should_stop():
    #         for i in range(200):
    #             example, l = sess.run([train_image, train_label])  # 在会话中取出image和label
    #             print('train:')
    #             print(np.shape(example))
    #             print(example)
    #             print(l)
    #             cv2.imshow('1',example[0])
    #             cv2.waitKey(0)
    #     except tf.errors.OutOfRangeError:
    #         print('Done reading')
    #     finally:
    #         coord.request_stop()
    #
    #     coord.request_stop()
    #     coord.join(threads)