'''
@inproceedings{eccv18refrmv,
  title={Seeing deeply and bidirectionally: a deep learning approach for single image reflection removal},
  author={Yang, Jie and Gong, Dong and Liu, Lingqiao and Shi, Qinfeng},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={654--669},
  year={2018}
}
'''
import tensorflow as tf
from tensorflow.keras import layers


###############################################################################
# Functions
###############################################################################
def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = layers.BatchNormalization(axis=1)  # Specify axis for channel-first format
    elif norm_type == 'instance':
        norm_layer = layers.LayerNormalization(axis=1)  # Specify axis for channel-first format
    else:
        print('Normalization layer [%s] is not found' % norm_type)
        norm_layer = layers.Lambda(lambda x: x)
    return norm_layer


##############################################################################
# Classes
##############################################################################
class Generator_cascade(tf.keras.Model):
    def __init__(self, input_nc=3, output_nc=3, ns=[7, 5, 5], ngf=64, norm='batch', use_dropout=False, iteration=0, padding_type='zero', upsample_type='transpose'):
        super(Generator_cascade, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.iteration = iteration
        norm_layer = norm

        self.model1 = UnetGenerator(input_nc, output_nc, ns[0], ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        self.model2 = UnetGenerator(input_nc * 2, output_nc, ns[1], ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        if self.iteration > 0:
            self.model3 = UnetGenerator(input_nc * 2, output_nc, ns[2], ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    def call(self, input, fromG2=False, pre_result=None):
        if not fromG2:
            x = self.model1(input)
            res = [x]
            for i in range(self.iteration + 1):
                if i % 2 == 0:
                    xy = tf.concat([x, input], axis=1)
                    z = self.model2(xy)
                    res += [z]
                else:
                    zy = tf.concat([z, input], axis=1)
                    x = self.model3(zy)
                    res += [x]
        else:
            res = [pre_result]
            for i in range(self.iteration + 1):
                if i % 2 == 0:
                    xy = tf.concat([pre_result, input], axis=1)
                    z = self.model2(xy)
                    res += [z]
                else:
                    zy = tf.concat([z, input], axis=1)
                    x = self.model3(zy)
                    res += [x]
        return res

class Generator_cascade_withEdge(tf.keras.Model):
    def __init__(self, input_nc=3, output_nc=3, ns=[7,5,5], ngf=64, norm='batch', use_dropout=False, iteration=0, padding_type='zero', upsample_type='transpose'):
        super(Generator_cascade_withEdge, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.iteration = iteration
        norm_layer = norm

        self.model1 = UnetGenerator(input_nc + 1, output_nc, ns[0], ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        self.model2 = UnetGenerator(input_nc * 2 + 1, output_nc, ns[1], ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        if self.iteration > 0:
            self.model3 = UnetGenerator(input_nc * 2 + 1, output_nc, ns[2], ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    def call(self, input, fromG2=False, pre_result=None):
        if fromG2 == False:
            x = self.model1(input)
            res = [x]
            for i in range(self.iteration + 1):
                if i % 2 == 0:
                    xy = tf.concat([x, input], axis=1)
                    z = self.model2(xy)
                    res += [z]
                else:
                    zy = tf.concat([z, input], axis=1)
                    x = self.model3(zy)
                    res += [x]
        else:
            res = [pre_result]
            for i in range(self.iteration + 1):
                if i % 2 == 0:
                    xy = tf.concat([pre_result, input], axis=1)
                    z = self.model2(xy)
                    res += [z]
                else:
                    zy = tf.concat([z, input], axis=1)
                    x = self.model3(zy)
                    res += [x]
        return res

class UnetGenerator(tf.keras.Model):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=None, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # Construct U-Net structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True, use_dropout=use_dropout)

        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer, outermost_input_nc=input_nc)

        self.model = unet_block

    def call(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(tf.keras.Model):
    def __init__(self, outer_nc, inner_nc, submodule=None, outermost=False, innermost=False, norm_layer='batch', use_dropout=False, outermost_input_nc=-1):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        submodule = submodule if submodule is not None else layers.Lambda(lambda x: x)

        if outermost and outermost_input_nc > 0:
            downconv = layers.Conv2D(inner_nc, kernel_size=4, strides=2, padding='same', data_format='channels_first')
        else:
            downconv = layers.Conv2D(inner_nc, kernel_size=4, strides=2, padding='same', data_format='channels_first')

        downrelu = layers.LeakyReLU(0.2)
        downnorm = get_norm_layer(norm_type=norm_layer)
        uprelu = layers.ReLU()
        upnorm = get_norm_layer(norm_type=norm_layer)

        if outermost:
            upconv = layers.Conv2DTranspose(outer_nc, kernel_size=4, strides=2, padding='same', data_format='channels_first')
            down = [downconv]
            up = [uprelu, upconv, layers.Activation('tanh')]
            model = down + [submodule] + up
        elif innermost:
            upconv = layers.Conv2DTranspose(outer_nc, kernel_size=4, strides=2, padding='same', data_format='channels_first')
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = layers.Conv2DTranspose(outer_nc, kernel_size=4, strides=2, padding='same', data_format='channels_first')
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [layers.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = tf.keras.Sequential(model)

    def call(self, x):
        x1 = self.model(x)
        diff_h = x.shape[2] - x1.shape[2]
        diff_w = x.shape[3] - x1.shape[3]
        x1 = tf.pad(x1, [[0, 0], [0, 0], [diff_h // 2, diff_h - diff_h // 2], [diff_w // 2, diff_w - diff_w // 2]])
        if self.outermost:
            return x1
        else:
            return tf.concat([x1, x], axis=1)  # Concatenate along the channels axis
