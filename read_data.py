import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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
train_image_batch, train_label_batch = tf.train.batch([decoded_image, decoded_label], batch_size=batch_size, capacity=3*batch_size)
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(5):
        mpi_depth, ref_depth = sess.run([train_image_batch, train_label_batch])
        print(mpi_depth.shape)
        print(ref_depth.shape)
        difference = np.abs(np.subtract(mpi_depth, ref_depth))
        print("average_error: ", np.mean(difference))
        # plt.subplot(1, 2, 1)
        # im = plt.imshow(mpi_depth[0, :, :, 0], cmap='viridis')
        # plt.colorbar(im)
        # plt.subplot(1, 2, 2)
        # ref = plt.imshow(ref_depth[0, :, :, 0], cmap='viridis')
        # plt.colorbar(ref)
        # plt.show()

    coord.request_stop()
    coord.join(threads)




# f = h5py.File('DeepToF_validation_1.7k_256x256.h5', 'r')
# print(f.keys())
# # print(f['albedo_id'])
# # print(f['amps'])
# # amps = np.array(f['amps'])
# # print(amps.shape)
# # test = amps[0]
# # print(test.shape)
# # a = np.transpose(test, [1, 2, 0])
# # b = a[:, :, 0]
# # # b = np.array([a[:, :, 0] for _ in range(3)]).transpose([1, 2, 0])
# # # print(b.shape)
# # # print(b)
# # im = plt.imshow(b*20, cmap='viridis')
# # plt.colorbar(im)
# # plt.show()
# depth = np.array(f['depth'])
# depth_ref = np.array(f['depth_ref'])
# print(depth.shape)
# print(depth_ref.shape)
# test = depth[0, 0, :, :]
# print(test)
# im = plt.imshow(test, cmap='viridis')
# plt.colorbar(im)
# plt.show()
# ref = depth_ref[0, 0, :, :]
# print(ref)
# im_ref = plt.imshow(ref, cmap='viridis')
# plt.colorbar(im_ref)
# plt.show()