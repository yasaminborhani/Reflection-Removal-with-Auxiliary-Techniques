import tensorflow as tf
from tensorflow.keras import layers, models

class similarity_VGG_bn(tf.keras.Model):
    def __init__(self, input_channels=3):
        super(similarity_VGG_bn, self).__init__()

        vgg_bn = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        self.cnn_bn = tf.keras.Model(inputs=vgg_bn.input, outputs=vgg_bn.layers[1].output)

        self.conv1 = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', input_shape=(224, 224, input_channels))
        self.bn1 = layers.BatchNormalization(epsilon=1e-5, momentum=0.1)
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.bn2 = layers.BatchNormalization(epsilon=1e-5, momentum=0.1)
        self.relu2 = layers.ReLU()
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        self.conv3 = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')
        self.bn3 = layers.BatchNormalization(epsilon=1e-5, momentum=0.1)
        self.relu3 = layers.ReLU()

        self.conv4 = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')
        self.bn4 = layers.BatchNormalization(epsilon=1e-5, momentum=0.1)
        self.relu4 = layers.ReLU()
        self.pool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

        self.conv5 = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')
        self.bn5 = layers.BatchNormalization(epsilon=1e-5, momentum=0.1)
        self.relu5 = layers.ReLU()

    def init_from_vgg19_bn(self):
        vgg_layers = self.cnn_bn.layers
        sim_layers = self.layers
        for i, (sim_layer, vgg_layer) in enumerate(zip(sim_layers, vgg_layers)):
            if isinstance(sim_layer, layers.Conv2D) and i != 0:
                # Ensure that the custom layer is built
                sim_layer.build((None, 224, 224, 3))
                sim_layer.set_weights(vgg_layer.get_weights())

    def call(self, x):
        outputs = []

        out = self.conv1(x)
        outputs.append(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        outputs.append(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        outputs.append(out)
        out = self.bn3(out)
        out = self.relu3(out)

        out = self.conv4(out)
        outputs.append(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.pool4(out)

        out = self.conv5(out)
        outputs.append(out)
        out = self.bn5(out)
        out = self.relu5(out)

        return outputs
