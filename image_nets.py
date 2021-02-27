import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class ImageNet:
    def __init__(self, dataset, epochs=10, num_classes=10):
        self.epochs = epochs
        self.dataset = dataset
        self.num_classes = num_classes
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.image_height = None
        self.image_width = None
        self.image_channels = None
        self.model = None
        self.probability_model = None


    def loadDataset(self):
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.dataset.load_data()
        self.image_height = self.train_images.shape[1]
        self.image_width = self.train_images.shape[2]
        if (len(self.train_images.shape) == 4):
            self.image_channels = self.train_images.shape[3]

    def getModelSummary(self):
        self.model.summary()

    def normalize(self):
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0


    def makeModel(self):
        self.model = tf.keras.Sequential()
        if self.image_channels!=None:
            self.model.add(tf.keras.layers.Flatten(input_shape=(self.image_height, self.image_width, self.image_channels)))
        else:
            self.model.add(tf.keras.layers.Flatten(input_shape=(self.image_height, self.image_width)))

        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.num_classes))


    def compileModel(self):
        self.model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


    def fit(self):
        self.model.fit(self.train_images, self.train_labels, epochs=self.epochs)


    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(self.test_images, self.test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)


    def predict(self):
        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        predictions = self.probability_model.predict(self.test_images)
        print(predictions[0])
        print(np.argmax(predictions[0]))


    def run(self):
        self.loadDataset()
        self.normalize()
        self.makeModel()
        self.compileModel()
        self.fit()
        self.evaluate()
        self.predict()




fashion_mnist = tf.keras.datasets.fashion_mnist
cifar10 = tf.keras.datasets.cifar10
mnist = tf.keras.datasets.mnist

nn = ImageNet(cifar10)
nn.run()
nn.getModelSummary()