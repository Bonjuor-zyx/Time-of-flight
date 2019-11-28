import tensorflow as tf
import numpy as np
import sample_net

def train():
    x = tf.placeholder(tf.float32, [None, 256, 256, 1])  # channels_first
    y_ = tf.placeholder(tf.float32, [None, 256, 256, 1])

    global_step = tf.Variable(0, trainable=False)
    boundaries = [10000, 15000, 20000, 25000]
    values = [0.1, 0.05, 0.01, 0.005, 0.001]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    weight_decay = 2e-4

    batch_size = 16

    train_filename_queue = tf.train.string_input_producer(['train.tfrecords'])
    train_reader = tf.TFRecordReader()
    _, serialized_example = train_reader.read(train_filename_queue)
    train_features = tf.parse_single_example(serialized_example,
                                             features={
                                                 'image': tf.FixedLenFeature([], tf.string),
                                                 'label': tf.FixedLenFeature([], tf.string),
                                             })
    image_raw, label = train_features['image'], train_features['label']
    decoded_image = tf.decode_raw(image_raw, tf.float32)
    decoded_image = tf.reshape(decoded_image, [256, 256, 1])
    decoded_label = tf.decode_raw(label, tf.float32)
    decoded_label = tf.reshape(decoded_label, [256, 256, 1])
    train_image_batch, train_label_batch = tf.train.batch([decoded_image, decoded_label], batch_size=batch_size,
                                                          capacity=3 * batch_size)

    y = sample_net.inference(x, True)
    difference = tf.abs(tf.subtract(y, y_))
    MAE = tf.reduce_mean(difference)
    l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
    loss = MAE + l2_loss
    optimizer = tf.train.AdamOptimizer(1e-3)

    updata_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(updata_op):
        opt_op = optimizer.minimize(loss, global_step)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(30000):#training steps
            xs, ys = sess.run([train_image_batch, train_label_batch])
            original_loss = np.abs(np.subtract(xs, ys))
            original_mae = np.mean(original_loss)
            _, loss_value, mae = sess.run([opt_op, loss, MAE], feed_dict={x: xs, y_: ys})
            if i % 100 == 0:
                print("After {} training steps, loss is {}, the bath mae is {}, the original mae is{}.".format(i, loss_value, mae, original_mae))
            if i % 5000 == 0:
                #print("After {} training steps, test accuracy is {}".format(i, loss_value))
                saver.save(sess, 'model/my-model', global_step=i)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()