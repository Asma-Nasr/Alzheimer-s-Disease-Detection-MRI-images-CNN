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
    
    def fit(self, train_generator, validation_data=None, epochs=10, steps_per_epoch=None, validation_steps=None):
        history = self.model.fit(
            train_generator,
            validation_data=validation_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps
        )
        return history
    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x, y):
        return self.model.evaluate(x,y)

    def save(self, filepath):
        self.model.save(filepath)

class VGG16Model(BaseModel):
    def create_model(self):
        vgg = VGG16(input_shape=self.input_shape, weights='imagenet', include_top=False)
        for layer in vgg.layers:
            layer.trainable = False

        x = Flatten()(vgg.output)
        prediction = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=vgg.input, outputs=prediction)
        return model


class VGG19Model(BaseModel):
    def create_model(self):
        vgg = VGG19(input_shape=self.input_shape, weights='imagenet', include_top=False)
        for layer in vgg.layers:
            layer.trainable = False

        x = Flatten()(vgg.output)
        prediction = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=vgg.input, outputs=prediction)
        return model

class ResNet50Model(BaseModel):
    def create_model(self):
        resnet = ResNet50(input_shape=self.input_shape, weights='imagenet', include_top=False)
        for layer in resnet.layers:
            layer.trainable = False

        x = Flatten()(resnet.output)
        prediction = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=resnet.input, outputs=prediction)
        return model
