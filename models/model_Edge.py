import tensorflow as tf
from tensorflow.keras import layers, Model

class Edge_UNet(tf.keras.Model):
    def __init__(self):
        super(Edge_UNet, self).__init__()

        # Downsample layers
        self.downsample1 = tf.keras.Sequential([
            layers.Conv2D(64, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.Conv2D(64, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first')
        ])

        self.downsample2 = tf.keras.Sequential([
            layers.Conv2D(128, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.Conv2D(128, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first')
        ])

        self.downsample3 = tf.keras.Sequential([
            layers.Conv2D(256, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.Conv2D(256, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.Conv2D(256, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first')
        ])

        self.downsample4 = tf.keras.Sequential([
            layers.Conv2D(512, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.Conv2D(512, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.Conv2D(512, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first')
        ])

        self.downsample5 = tf.keras.Sequential([
            layers.Conv2D(512, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.Conv2D(512, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.Conv2D(512, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2)
        ])

        # Upsample layers
        self.upsample1 = tf.keras.Sequential([
            layers.Conv2DTranspose(512, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(512, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(512, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2)
        ])

        self.upsample2 = tf.keras.Sequential([
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear', data_format='channels_first'),
            layers.Conv2DTranspose(512, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(512, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(256, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2)
        ])

        self.upsample3 = tf.keras.Sequential([
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear', data_format='channels_first'),
            layers.Conv2DTranspose(256, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(256, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(128, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2)
        ])

        self.upsample4 = tf.keras.Sequential([
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear', data_format='channels_first'),
            layers.Conv2DTranspose(128, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(64, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2)
        ])

        self.upsample5 = tf.keras.Sequential([
            layers.UpSampling2D(size=(2, 2), interpolation='bilinear', data_format='channels_first'),
            layers.Conv2DTranspose(64, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(64, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.BatchNormalization(axis=1),
            layers.LeakyReLU(0.2)
        ])

        self.final_output = tf.keras.Sequential([
            layers.Conv2DTranspose(1, (3, 3), strides=1, padding='same', use_bias=True, data_format='channels_first'),
            layers.Activation('sigmoid')
        ])

    def call(self, x):
        x = self.downsample1(x)
        skip1 = x
        x = self.downsample2(x)
        skip2 = x
        x = self.downsample3(x)
        skip3 = x
        x = self.downsample4(x)
        skip4 = x
        x = self.downsample5(x)

        x = self.upsample1(x)
        x = x + skip4
        x = self.upsample2(x)
        x = x + skip3
        x = self.upsample3(x)
        x = x + skip2
        x = self.upsample4(x)
        x = x + skip1
        x = self.upsample5(x)
        x = self.final_output(x)
        return x
