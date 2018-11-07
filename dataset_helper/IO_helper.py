import cv2
import numpy as np
import os
import random
import copy
from labels import *

# 图片资源根目录，具体应用时，需修改到本地相对应的路径
date_resources_path = '../datasets/'
# 训练集图片目录
train_path = 'train/'
# 测试集图片目录
test_path = 'test/'
# 验证集图片目录
val_path = 'val/'
# opencv 人脸检测
face_patterns = cv2.CascadeClassifier(
        '../res/haarcascade_frontalface_default.xml')

# 基于opencv进行人脸检测
# 返回人脸图片及坐标
# 默认的size 为42*42
def detectFace(img, size = (42, 42)):
    pass

class ImageObject:
    def __init__(self, data, label = -1):
        self.data = data
        self.label = label

    # 将图片归一化到[0,1]区间
    def encode_image(self, size = image_shape):
        # self.data = cv2.resize(self.data, image_shape, interpolation=cv2.INTER_CUBIC)
        # 这里/255是为了将像素值归一化到[0，1]
        self.data = self.data / 255.
        self.data = self.data.astype(np.float32)
        # self.data = np.reshape(self.data, (1, image_shape[0]*image_shape[1]))

    @staticmethod
    def encode(data, size=image_shape):
        img = cv2.resize(data, image_shape_2, interpolation=cv2.INTER_CUBIC)
        # img = data
        # 这里/255是为了将像素值归一化到[0，1]
        img = img / 255.
        img = img.astype(np.float32)
        return img
        # self.data = np.reshape(self.data, (1, image_shape[0]*image_shape[1]))

    # 恢复图片
    def decode_image(self, size = image_shape):
        pass

    # @staticmethod
    # def get_face(image):
    # 镜像图片
    @staticmethod
    def reverse_face(face):
        h, w = face.shape[0 : 2]
        res = copy.deepcopy(face)
        for i in  range(h):
            for j in range(w):
                res[i, j] = face[i][w - j - 1]
        return res

    # 偏移图像， 取各角
    @staticmethod
    def crop_face(face, win = (36,36)):
        w, h = face.shape[0 : 2]
        face_lt = cv2.resize(face[0 : win[1], 0 : win[0]], (42,42))
        face_rt = cv2.resize(face[0 : win[1], w - win[0]: w], (42, 42))
        face_lb = cv2.resize(face[h - win[1] : h, 0: win[0]], (42, 42))
        face_rb = cv2.resize(face[h - win[1] : h, w - win[0]: w], (42, 42))
        # face_lt = face[0: win[1], 0: win[0]]
        # face_rt = face[0: win[1], w - win[0]: w]
        # face_lb = face[h - win[1]: h, 0: win[0]]
        # face_rb = face[h - win[1]: h, w - win[0]: w]
        return [face, face_lt, face_rt, face_lb, face_rb]
    # 给图片增加噪声
    @staticmethod
    def noiseing(img):
        # img = cv2.cvtColor(rgbimg, cv2.COLOR_BGR2GRAY)
        param = 30
        grayscale = 256
        w = img.shape[1]
        h = img.shape[0]
        newimg = np.zeros((h, w, 3), np.uint8)
        # row and col
        for x in range(0, h):
            for y in range(0, w, 2):  # Avoid exceeding boundaries
                r1 = np.random.random_sample()
                r2 = np.random.random_sample()
                z1 = param * np.cos(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))
                z2 = param * np.sin(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))

                fxy_0 = int(img[x, y, 0] + z1)
                fxy_1 = int(img[x, y, 1] + z1)
                fxy_2 = int(img[x, y, 2] + z1)
                fxy1_0 = int(img[x, y + 1, 0] + z2)
                fxy1_1 = int(img[x, y + 1, 1] + z2)
                fxy1_2 = int(img[x, y + 1, 2] + z2)
                # f(x,y)
                if fxy_0 < 0:
                    fxy_val_0 = 0
                elif fxy_0 > grayscale - 1:
                    fxy_val_0 = grayscale - 1
                else:
                    fxy_val_0 = fxy_0
                if fxy_1 < 0:
                    fxy_val_1 = 0
                elif fxy_1 > grayscale - 1:
                    fxy_val_1 = grayscale - 1
                else:
                    fxy_val_1 = fxy_1
                if fxy_2 < 0:
                    fxy_val_2 = 0
                elif fxy_2 > grayscale - 1:
                    fxy_val_2 = grayscale - 1
                else:
                    fxy_val_2 = fxy_2
                # f(x,y+1)
                if fxy1_0 < 0:
                    fxy1_val_0 = 0
                elif fxy1_0 > grayscale - 1:
                    fxy1_val_0 = grayscale - 1
                else:
                    fxy1_val_0 = fxy1_0
                if fxy1_1 < 0:
                    fxy1_val_1 = 0
                elif fxy1_1 > grayscale - 1:
                    fxy1_val_1 = grayscale - 1
                else:
                    fxy1_val_1 = fxy1_1
                if fxy1_2 < 0:
                    fxy1_val_2 = 0
                elif fxy1_2 > grayscale - 1:
                    fxy1_val_2 = grayscale - 1
                else:
                    fxy1_val_2 = fxy1_2

                newimg[x, y, 0] = fxy_val_0
                newimg[x, y, 1] = fxy_val_1
                newimg[x, y, 2] = fxy_val_2
                newimg[x, y + 1, 0] = fxy1_val_0
                newimg[x, y + 1, 1] = fxy1_val_1
                newimg[x, y + 1, 2] = fxy1_val_2
        return newimg

    # 增强图片，对图像进行扩展
    @staticmethod
    def enhance_image(images):
        res = []
        for image in images:
            target = ImageObject.crop_face(image)
            for img in target:
                res.append(img)
                res.append(ImageObject.reverse_face(img))
        return res


class ImageDataResource:
    def __init__(self):
        # 图片总数
        self.image_sum = 0
        # 分类图片数量
        self.kind_count = []
        # 单张图片形状
        self.image_shape = []
        # 图片列表，二维列表，0-6下标作分类
        self.data = []
        # 乱序的数据，整合为一个数组，是ImageObject的集合
        self.un_seq_data = []

    # 获取图片的形状
    def init_shape(self):
        self.image_shape = np.shape(self.data[0][0])

    # 打乱数据
    def shuffle_data(self):
        for i in range(7):
            for j in range(self.kind_count[i]):
                image = ImageObject(self.data[i][j],i)
                self.un_seq_data.append(image)
        random.shuffle(self.un_seq_data)


# 获取指定目录下的全部图片，并按分类整合
# 返回值单个ImageDataResource
def read_images(path):
    resource = ImageDataResource()
    # 按照表情类别遍历文件夹
    for kind in range(7):
        root_path = date_resources_path + path + str(kind)
        kind_num = 0
        images = []
        for image in os.listdir(root_path):
            resource.image_sum = resource.image_sum + 1
            kind_num = kind_num + 1
            images.append(cv2.imread(root_path + '/' +image, 0))
        resource.kind_count.append(kind_num)
        resource.data.append(images)
    resource.init_shape()
    return resource

# 同上述方法，并对数据增强
def read_images_enhance(path):
    resource = ImageDataResource()
    # 按照表情类别遍历文件夹
    for kind in range(7):
        root_path = date_resources_path + path + str(kind)
        kind_num = 0
        images = []
        for image in os.listdir(root_path):
            resource.image_sum = resource.image_sum + 2
            kind_num = kind_num + 2
            face = cv2.imread(root_path + '/' +image, 0)
            face_reverse = ImageObject.reverse_face(face)
            # faces = ImageObject.enhance_image([face])
            images.append(face)
            images.append(face_reverse)
        resource.kind_count.append(kind_num)
        resource.data.append(images)
    resource.init_shape()
    return resource

# 同上述方法，并对数据增强，并入测试集
def read_images_enhance_with_test(path, path_test):
    resource = ImageDataResource()
    # 按照表情类别遍历文件夹
    for kind in range(7):
        root_path = date_resources_path + path + str(kind)
        kind_num = 0
        images = []
        for image in os.listdir(root_path):
            resource.image_sum = resource.image_sum + 2
            kind_num = kind_num + 2
            face = cv2.imread(root_path + '/' +image, 0)
            face_reverse = ImageObject.reverse_face(face)
            # faces = ImageObject.enhance_image([face])
            images.append(face)
            images.append(face_reverse)
        resource.kind_count.append(kind_num)
        resource.data.append(images)

    # 按照表情类别遍历文件夹
    for kind in range(7):
        root_path = date_resources_path + path_test + str(kind)
        kind_num = 0
        for image in os.listdir(root_path):
            resource.image_sum = resource.image_sum + 2
            kind_num = kind_num + 2
            face = cv2.imread(root_path + '/' + image, 0)
            face_reverse = ImageObject.reverse_face(face)
            # faces = ImageObject.enhance_image([face])
            resource.data[kind].append(face)
            resource.data[kind].append(face_reverse)
        resource.kind_count[kind] = resource.kind_count[kind] + kind_num
    resource.init_shape()
    return resource


# 获取训练图片，返回值形式同read_images()
def read_train_images():
    return read_images(train_path)

def read_enhance_train_images():
    return read_images_enhance(train_path)

def read_enhace_train_with_test_images():
    return read_images_enhance_with_test(train_path, test_path)

# 获取测试图片，返回值形式同read_images()
def read_test_images():
    return read_images(test_path)


# 获取验证图片，返回值形式同read_images()
def read_val_images():
    return read_images(val_path)

if __name__ == '__main__':
    # pass
    # resource = read_train_images()
    # resource.shuffle_data()
    # print(resource.image_sum)
    # print(len(resource.un_seq_data))
    # print(np.shape(resource.data[0]))
    img = cv2.imread('E:/res/img/31.jpg', 1)
    newimg = ImageObject.reverse_face(img)
    cv2.imwrite('E:/res/img/32.jpg', newimg)
    # for i in range(len(resource.un_seq_data)):
    #     print(resource.un_seq_data[i].label)
    #
    # cv2.imshow('1',resource.un_seq_data[0].data)
    # cv2.imshow('2', resource.un_seq_data[len(resource.un_seq_data) - 1].data)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # image = cv2.imread('E:/tmp/emote-recognition/datasets/train/0/00000.jpg',0)
    # data = ImageObject(image, 0)
    # data.encode_image()
    # print(data.data)
    # print(np.shape(data.data))
    # cv2.imshow('data',np.reshape(data.data,(48,48)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # images = ImageObject.enhance_image([image])
    # for i in range(10):
    #     cv2.imshow(str(i), images[i])
    #     cv2.resizeWindow(str(i), 640, 480);
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()