import tensorflow as tf
import numpy as np
import h5py

def load(filname):
    data = h5py.File(filname, 'r')

    return data

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


#<KeysViewHDF5 ['albedo_id', 'amps', 'depth', 'depth_ref', 'mpi_abs', 'mpi_rel', 'scene_id']>
data_file = 'DeepToF_validation_1.7k_256x256.h5'
tf_file_name = 'train.tfrecords'
writer = tf.python_io.TFRecordWriter(tf_file_name)
data = load(data_file)
print(data.keys())
num = len(data['albedo_id'])
for i in range(num):
    label = np.array(data['depth_ref'][i])
    label = np.transpose(label, [1, 2, 0])
    MPI_depth = np.array(data['depth'][i])
    MPI_depth = np.transpose(MPI_depth, [1, 2, 0])
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _bytes_feature(label.tostring()),
        'image': _bytes_feature(MPI_depth.tostring())
    }))
    writer.write(example.SerializeToString())
writer.close()



