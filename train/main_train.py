import os
import tensorflow as tf
import numpy as np
from inference.main_inference import inference
from dataset_helper.TFRecords_helper import *

BATCH_SIZE = 50
LEARNING_RATE = 0.0001
TRAINING_STEPS = 100000

MODEL_SAVE_PATH = "../models/model.ckpt"


def train(path = train_recorder_enhance_path):
    filename_queue = tf.train.string_input_producer([path], num_epochs=None)  # 读入流中

    train_image, train_label = decode_from_tf_records(filename_queue, is_batch=True, shape= image_shape_2_)

    x = tf.placeholder(tf.float32, [None, 2304], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, 7], name="y-input")
    x_image = tf.reshape(x, [-1, 48, 48, 1])
    y = inference(x_image, True)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.control_dependencies([optimizer, accuracy]):
        train_op = tf.no_op(name="train")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            for i in  range(TRAINING_STEPS):
                xs, ys = sess.run([train_image, train_label])
                xs_, ys_ = modify_size(xs,ys,shape=[2304])
                if i % 50 == 0:
                    acc = sess.run(accuracy, feed_dict={x:xs_, y_:ys_})
                    print("step by %d , training accuracy %g"%(i, acc))

                _, cost_val, optimizer_val = sess.run([train_op, cost, optimizer], feed_dict={x : xs_, y_ : ys_})
                # print(cost_val)

            # test_image, test_label = decode_from_tf_records(filename_queue, is_batch=True)
            # for i in range(100):
            #     xs, ys = sess.run([test_image, test_label])
            #     xs_, ys_ = modify_size(xs, ys)
            #     acc = sess.run(accuracy, feed_dict={x: xs_, y_: ys_})
            #     print("test step by %d, testing accuracy %g"%(i, acc))
            saver.save(sess, MODEL_SAVE_PATH)

        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)
    # writer = tf.summary.FileWriter("log/train.log", tf.get_default_graph())
    # writer.close()

def main(args = None):
    train()

if __name__ == '__main__':
    tf.app.run()

