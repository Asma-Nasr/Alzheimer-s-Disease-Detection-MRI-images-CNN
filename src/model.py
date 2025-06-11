import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, SeparableConv2D, BatchNormalization
from tensorflow.keras.applications import VGG16, VGG19, ResNet50
from tensorflow.keras.models import Model

class BaseModel:
    def __init__(self, input_shape=(150, 150, 3), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.create_model()

    def create_model(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss=tf.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.AUC(name='auc')]
        )

    def summary(self):
        return self.model.summary()
