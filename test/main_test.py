from inference.main_inference import inference
from dataset_helper.TFRecords_helper import *
from dataset_helper.count_result import *
import tensorflow as tf
import time

BATCH_SIZE = 50

MODEL_SAVE_PATH = os.getcwd() + "/models/model.ckpt"

def test(path = val_recorder_path):
    filename_queue = tf.train.string_input_producer([path], num_epochs=None)  # 读入流中

    test_image, test_label = decode_from_tf_records(filename_queue, is_batch=True, shape= image_shape_2_)

    x = tf.placeholder(tf.float32, [None, 2304], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, 7], name="y-input")
    x_image = tf.reshape(x, [-1, 48, 48, 1])
    y = inference(x_image, False)
    prediction = tf.argmax(y, 1)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_result = TestResult()
    with tf.control_dependencies([accuracy]):
        test_op = tf.no_op(name="test")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, MODEL_SAVE_PATH)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        acc_sum = 0
        count_sum = 0
        try:
            for i in range(1000):
                xs, ys = sess.run([test_image, test_label])
                xs_, ys_ = modify_size(xs, ys, shape=[2304])
                _, acc, pred = sess.run([test_op, accuracy, prediction], feed_dict={x: xs_, y_: ys_})
                # print(ys)
                # print(pred)
                test_result.update(ys, pred)
                acc_sum = acc_sum + 50 * acc
                count_sum = count_sum + 50
                print("step by %d , training accuracy %g" % (i, acc))
            print("accuracy : ", acc_sum / count_sum)
            test_result.display()
            # test_image, test_label = decode_from_tf_records(filename_queue, is_batch=True)
            # for i in range(100):
            #     xs, ys = sess.run([test_image, test_label])
            #     xs_, ys_ = modify_size(xs, ys)
            #     acc = sess.run(accuracy, feed_dict={x: xs_, y_: ys_})
            #     print("test step by %d, testing accuracy %g"%(i, acc))

        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)

def app_test():
    x_ = tf.placeholder(tf.float32, [None, 2304], name="x-input")
    x_image = tf.reshape(x_, [-1, 48, 48, 1])
    y_ = inference(x_image, False)
    prediction = tf.argmax(y_, 1)
    with tf.control_dependencies([prediction]):
        test_op = tf.no_op(name="test")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, MODEL_SAVE_PATH)
        capture = cv2.VideoCapture(0)
        while (31):
            ret, frame = capture.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            coors = face_detect(gray_frame)
            if len(coors) == 0:
                continue
            faces = []
            for coor in coors:
                x, y, w, h = coor
                faces.append(ImageObject.encode(gray_frame[y : y + h, x : x + w], image_shape_2))
            # face = ImageObject(gray_frame)
            # face.encode_image(size=image_shape_2)
            face_target, _ = modify_size(faces, [], [2304])
            # infer, pred = sess.run([y, prediction], feed_dict={face_target)})
            _, infer , pred = sess.run([test_op, y_, prediction], feed_dict={x_: face_target})
            print(infer, pred)
            emotes = []
            for res in pred:
                emotes.append(emote_labels[res])
            frame = mark_human_emote(frame, coors, emotes)
            cv2.imshow("capture", frame)
            if cv2.waitKey(31) & 0xFF == ord('q'):
                break
            time.sleep(0.1)
        cv2.destroyAllWindows()

def recognize_single_image(image):
    x_ = tf.placeholder(tf.float32, [None, 2304], name="x-input")
    x_image = tf.reshape(x_, [-1, 48, 48, 1])
    inference_y = inference(x_image, False)
    prediction = tf.argmax(inference_y, 1)
    with tf.control_dependencies([prediction]):
        test_op = tf.no_op(name="test")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, MODEL_SAVE_PATH)
        coors = face_detect(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = []
        for coor in coors:
            x, y, w, h = coor
            faces.append(ImageObject.encode(gray_image[y: y + h, x: x + w], image_shape_2))
        data, _ = modify_size(faces, [], [2304])
        print(np.shape(data))
        _, infer, pred = sess.run([test_op, inference_y, prediction], feed_dict={x_: data})
        print(infer, pred)
        emotes = []
        for res in pred:
            emotes.append(emote_labels[res])
        frame = mark_human_emote(image, coors, emotes)
        cv2.imshow("display", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main(args = None):
    # app_test()
    test()

if __name__ == '__main__':
    tf.app.run()

