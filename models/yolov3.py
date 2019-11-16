from model_zoo import Model
import tensorflow as tf


class YOLOV3Model(Model):
    """
    YOLO V3 Model
    """
    
    def __init__(self, config):
        """
        init layers
        :param config:
        """
        super(YOLOV3Model, self).__init__(config)
    
    def conv_block(self, inputs, filters, kernel_size, strides=(1, 1)):
        """
        Convolutional block.
        :param inputs: inputs
        :param filters: convolutional filters
        :param kernel_size: kernel size
        :param strides: stride, default to (1, 1)
        :param padding: padding, default to `same`, if stride is (2, 2), change to `valid`
        :return: convolutional result
        """
        padding = 'valid' if strides == (2, 2) else 'same'
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   strides=strides, padding=padding,
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                                   use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    
    def res_block(self, inputs, filters, iteration):
        """
        Res block.
        :param inputs: inputs
        :param filters: res block filters
        :param iteration: iteration count
        :return: res result
        """
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        x = self.conv_block(x, filters=filters, kernel_size=(3, 3), strides=(2, 2))
        for _ in range(iteration):
            y = self.conv_block(x, filters=filters // 2, kernel_size=(1, 1))
            y = self.conv_block(y, filters=filters, kernel_size=(3, 3))
            x = tf.keras.layers.Add()([x, y])
        return x
    
    def dark_block(self, inputs):
        """
        Darknet block
        :param inputs: inputs
        :return: darknet result
        """
        x = self.conv_block(inputs, filters=32, kernel_size=(3, 3))
        x = self.res_block(x, filters=64, iteration=1)
        x = self.res_block(x, filters=128, iteration=2)
        x = self.res_block(x, filters=256, iteration=8)
        x = self.res_block(x, filters=512, iteration=8)
        x = self.res_block(x, filters=1024, iteration=4)
        return x
    
    def inputs(self):
        return tf.keras.Input(shape=(416, 416, 3))
    
    def outputs(self, inputs):
        outputs = self.dark_block(inputs)
        return outputs
