import os
import sys
import cv2
import numpy as np
import requests
from test.main_test import test,app_test, recognize_single_image
from train.main_train import train

def setup():
    # os.system('pip install -r requirements.txt')
    os.system('mkdir models')
    print('create dir models')
    os.system('mkdir datasets')
    print('create dir datasets')
    # # download model
    # print('start download model')
    # print('start download checkpoint')
    # response = requests.get('http://tomax.xin/models/checkpoint')
    # with open('models/checkpoint', 'wb') as f:
    #     f.write(response.content)
    # print('success download checkpoint')
    # print('start download model.ckpt.data-00000-of-00001')
    # response = requests.get('http://tomax.xin/models/model.ckpt.data-00000-of-00001')
    # with open('models/model.ckpt.data-00000-of-00001', 'wb') as f:
    #     f.write(response.content)
    # print('success download model.ckpt.data-00000-of-00001')
    # print('start download model.ckpt.index')
    # response = requests.get('http://tomax.xin/models/model.ckpt.index')
    # with open('models/model.ckpt.index', 'wb') as f:
    #     f.write(response.content)
    # print('success download model.ckpt.index')
    # print('start download model.ckpt.meta')
    # response = requests.get('http://tomax.xin/models/model.ckpt.meta')
    # with open('models/model.ckpt.meta', 'wb') as f:
    #     f.write(response.content)
    # print('success download model.ckpt.meta')
    # print('success download model')
    # # download datasets
    # print('start download dataset')
    # print('start download train.tfrecords')
    # response = requests.get('http://tomax.xin/datasets/train.tfrecords')
    # with open('models/train.tfrecords', 'wb') as f:
    #     f.write(response.content)
    # print('success download train.tfrecords')
    # print('start download train_enhance.tfrecords')
    # response = requests.get('http://tomax.xin/models/train_enhance.tfrecords')
    # with open('models/train_enhance.tfrecords', 'wb') as f:
    #     f.write(response.content)
    # print('success download train_enhance.tfrecords')
    # print('start download test.tfrecords')
    # response = requests.get('http://tomax.xin/models/test_tfrecords')
    # with open('models/test_tfrecords', 'wb') as f:
    #     f.write(response.content)
    # print('success download test.tfrecords')
    # print('start download val.tfrecords')
    # response = requests.get('http://tomax.xin/models/val_tfrecords')
    # with open('models/val_tfrecords', 'wb') as f:
    #     f.write(response.content)
    print('success download val.tfrecords')

def run_as_video():
    app_test()

def recognition_image(path):
    try:
        img = cv2.imread(path, 1)
        recognize_single_image(img)
    except Exception:
        print('invalid picture')
    print(path)


if __name__ == '__main__':
    if (len(sys.argv) == 1):
        print('setup -- init the environment ,download the model and dataset')
        print('run -v -- run app in video')
        print('run -p path -- run app to recognition picture and input the path of the image')
    elif (sys.argv[1] == 'setup'):
        # setup()
        pass
    elif (sys.argv[1] == 'run'):
        # run()
        if len(sys.argv) == 3 and sys.argv[2] == '-v':
            run_as_video()
        elif len(sys.argv) == 3 and sys.argv[2] == '-train':
            train(os.getcwd() + '/datasets/train_enhance.tfrecords')
        elif len(sys.argv) == 3 and sys.argv[2] == '-test':
            test(os.getcwd() + '/datasets/val.tfrecords')
        elif len(sys.argv) >= 3 and sys.argv[2] == '-p':
            if len(sys.argv) < 4:
                print('please input the path of the picture')
            else:
                recognition_image(sys.argv[3])
        else:
            print('invalid command')
    else:
        print('invalid command')