import tensorflow as tf
import numpy as np
import sample_net
import matplotlib.pyplot as plt
MODEL_SAVE_PATH = './model/'

def eval():
    x = tf.placeholder(tf.float32, [None, 256, 256, 1])  # channels_first
    y_ = tf.placeholder(tf.float32, [None, 256, 256, 1])

    y = sample_net.inference(x, False)
    mae = tf.reduce_mean(tf.abs(tf.subtract(y, y_)))

    batch_size = 16

    filename_queue = tf.train.string_input_producer(['test.tfrecords'])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.string)
                                       })
    image_raw, label = features['image'], features['label']
    decoded_image = tf.decode_raw(image_raw, tf.float32)
    decoded_image = tf.reshape(decoded_image, [256, 256, 1])
    decoded_label = tf.decode_raw(label, tf.float32)
    decoded_label = tf.reshape(decoded_label, [256, 256, 1])
    test_image_batch, test_label_batch = tf.train.batch([decoded_image, decoded_label], batch_size=batch_size,
                                                          capacity=3 * batch_size)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        for i in range(10):
            mpi_depth, ref_depth = sess.run([test_image_batch, test_label_batch])
            difference = np.abs(np.subtract(mpi_depth, ref_depth))
            print(" original average_error: ", np.mean(difference))
            filtered_depth, filtered_average_error = sess.run([y, mae], feed_dict={x: mpi_depth, y_: ref_depth})
            print('filtered average error: ', filtered_average_error)
            plt.subplot(1, 3, 1)
            im = plt.imshow(mpi_depth[0, :, :, 0], cmap='viridis')
            plt.title('mpi_depth')
            #plt.colorbar(im)
            plt.subplot(1, 3, 2)
            ref = plt.imshow(ref_depth[0, :, :, 0], cmap='viridis')
            plt.title('reference_depth')
            #plt.colorbar(ref)
            plt.subplot(1, 3, 3)
            fil = plt.imshow(filtered_depth[0, :, :, 0], cmap='viridis')
            plt.title('filtered_depth')
            #plt.colorbar(fil)
            plt.show()

        coord.request_stop()
        coord.join(threads)
if __name__ == '__main__':
    eval()