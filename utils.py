import bz2
import pickle
import numpy as np
import keras
from keras import backend as K
import tensorflow


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size, n_classes = 221, dim=(460,460), n_channels=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size, 3), dtype=np.float64)
        y = np.empty(self.batch_size, dtype=np.float64)


        for i, ID in enumerate(list_IDs_temp):
            with bz2.BZ2File('../dataset/arm2d_3dof_cached_%s.pkl.bz2' % str(ID).zfill(8),'rb') as file:
                sample = pickle.load(file)
                X[i,] = sample['image']
                # y[i] = sample['joints']
                # y[i] = self.labels[ID]
                y[i] = self.labels[ID][0]

        y_cat = keras.utils.to_categorical(y, num_classes=self.n_classes)

        # return X, y
        return X, y_cat


def get_test_set(num_samples, offset=0):
    x = np.empty((num_samples, 460, 460, 3))
    # y = np.empty((num_samples, 3))
    y = np.empty(num_samples)

    for i in range(num_samples):
        with bz2.BZ2File('../dataset/arm2d_3dof_cached_%s.pkl.bz2' % str(i+offset).zfill(8),'rb') as file:
            sample = pickle.load(file)
            x[i] = sample['image']
            # y[i] = sample['joints']
            y[i] = sample['joints'][0]
    y_cat = keras.utils.to_categorical(y, num_classes=221)
    return x,y_cat
