import numpy as np
from model_zoo import Model
import tensorflow as tf


class YOLOV3Model(Model):
    """
    YOLO V3 Model, structure of model: https://qiniu.cuiqingcai.com/2019-11-17-162549.jpg
    """
    
    def __init__(self, config):
        """
        Init layers
        :param config:
        """
        self.number_anchors = config.get('number_anchors')
        
        self.number_classes = config.get('number_classes')
        self.anchors = [[[1.25, 1.625], [2.0, 3.75], [4.125, 2.875]],
                        [[1.875, 3.8125], [3.875, 2.8125], [3.6875, 7.4375]],
                        [[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]]
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
    
    def conv_set_block(self, inputs, filters):
        """
        Convolutional set block.
        :param inputs:
        :param filters:
        :return:
        """
        x = self.conv_block(inputs, filters=filters, kernel_size=(1, 1))
        x = self.conv_block(x, filters=filters * 2, kernel_size=(3, 3))
        x = self.conv_block(x, filters=filters, kernel_size=(1, 1))
        x = self.conv_block(x, filters=filters * 2, kernel_size=(3, 3))
        return self.conv_block(x, filters=filters, kernel_size=(1, 1))
    
    def output_block(self, inputs, filters, output_dim):
        """
        Predict block
        :param inputs:
        :param filters:
        :param output_dim:
        :return:
        """
        x = self.conv_block(inputs, filters=filters * 2, kernel_size=(3, 3))
        return tf.keras.layers.Conv2D(filters=output_dim, kernel_size=(1, 1))(x)
    
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
        self.x1 = x
        x = self.res_block(x, filters=512, iteration=8)
        self.x2 = x
        x = self.res_block(x, filters=1024, iteration=4)
        return x
    
    def inputs(self):
        """
        Define inputs.
        :return:
        """
        return tf.keras.Input(shape=(416, 416, 3)),
    
    def outputs(self, inputs):
        """
        Define Darknet and its prediction.
        :param inputs:
        :return:
        """
        x = self.dark_block(inputs)
        
        # output_dim
        output_dim = self.number_anchors * (self.number_classes + 5)
        
        # output1
        x = self.conv_set_block(x, filters=512)
        y1_conv = self.output_block(x, filters=512, output_dim=output_dim)
        y1_pred = self.predict_block(y1_conv, stride=32, layer=0)
        
        x = self.conv_block(x, filters=256, kernel_size=(1, 1))
        x = tf.keras.layers.UpSampling2D(size=2)(x)
        x = tf.keras.layers.Concatenate()([x, self.x2])
        
        # output2
        x = self.conv_set_block(x, filters=256)
        y2_conv = self.output_block(x, filters=256, output_dim=output_dim)
        y2_pred = self.predict_block(y2_conv, stride=16, layer=1)
        
        x = self.conv_block(x, filters=128, kernel_size=(1, 1))
        x = tf.keras.layers.UpSampling2D(size=2)(x)
        x = tf.keras.layers.Concatenate()([x, self.x1])
        
        # output3
        x = self.conv_set_block(x, filters=128)
        y3_conv = self.output_block(x, filters=128, output_dim=output_dim)
        y3_pred = self.predict_block(y3_conv, stride=8, layer=2)
        
        return y1_conv, y1_pred, y2_conv, y2_pred, y3_conv, y3_pred
    
    def hook(self):
        """
        Debug function.
        :return:
        """
        pass
    
    def predict_block(self, inputs, stride, layer):
        """
        Convert conv result to predict result.
        :param inputs:
        :param stride:
        :param layer: 0-2 layer
        :return:
        """
        conv_shape = tf.shape(inputs)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        
        # output_size = 52
        
        conv_output = tf.reshape(inputs, (-1, output_size, output_size, 3, 5 + self.number_classes))
        
        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]  # center
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]  # width, height
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]
        
        # build grid
        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)
        
        # center of predict
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        # width, height of predict
        pred_wh = (tf.exp(conv_raw_dwdh) * self.anchors[layer]) * stride
        # concat above results of predict
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        
        # confidence of predict
        pred_conf = tf.sigmoid(conv_raw_conf)
        # probs of each class
        pred_prob = tf.sigmoid(conv_raw_prob)
        # concat them and return
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
    
    def loss(self):
        """
        Convert result of output_block [None, anchor_size, anchor_size, 18] to [None, anchor_size, anchor_size, 3, (5 + number_classes)]
        :param inputs:
        :return:
        """
        
        def bbox_iou(boxes1, boxes2):
            boxes1_area = boxes1[..., 2] * boxes1[..., 3]
            boxes2_area = boxes2[..., 2] * boxes2[..., 3]
            
            boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
            boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
            
            left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
            right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
            
            inter_section = tf.maximum(right_down - left_up, 0.0)
            inter_area = inter_section[..., 0] * inter_section[..., 1]
            union_area = boxes1_area + boxes2_area - inter_area
            
            return 1.0 * inter_area / union_area
        
        def bbox_giou(boxes1, boxes2):
            boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
            boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
            
            boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                                tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
            boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                                tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)
            
            boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
            boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
            
            left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
            right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
            
            inter_section = tf.maximum(right_down - left_up, 0.0)
            inter_area = inter_section[..., 0] * inter_section[..., 1]
            union_area = boxes1_area + boxes2_area - inter_area
            iou = inter_area / union_area
            
            enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
            enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
            enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
            enclose_area = enclose[..., 0] * enclose[..., 1]
            giou = iou - 1.0 * (enclose_area - union_area) / enclose_area
            
            return giou
        
        def layer_loss(y_conv, y_pred, y_true, layer):
            """
            Compute
            :param y_conv:
            :param y_pred:
            :param y_true:
            :param i:
            :return:
            """
            bboxes = []
            conv_shape = tf.shape(y_conv)
            batch_size = conv_shape[0]
            output_size = conv_shape[1]
            input_size = STRIDES[layer] * output_size
            y_conv = tf.reshape(y_conv, (batch_size, output_size, output_size, 3, 5 + self.number_classes))
            
            conv_raw_conf = y_conv[:, :, :, :, 4:5]
            conv_raw_prob = y_conv[:, :, :, :, 5:]
            
            pred_xywh = y_pred[:, :, :, :, 0:4]
            pred_conf = y_pred[:, :, :, :, 4:5]
            
            label_xywh = y_true[:, :, :, :, 0:4]
            respond_bbox = y_true[:, :, :, :, 4:5]
            label_prob = y_true[:, :, :, :, 5:]
            
            giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
            input_size = tf.cast(input_size, tf.float32)
            bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
            
            # compute giou loss
            giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
            
            iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
            max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
            
            respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < 0.45, tf.float32)
            
            conf_focal = tf.pow(respond_bbox - pred_conf, 2)
            
            conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            )
            
            prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
            
            giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
            conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
            prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))
            
            return giou_loss, conf_loss, prob_loss
        
        def total_loss(y_true, y_output):
            """
            Compute total loss
            :param y_true:
            :param y_output:
            :return:
            """
            total_loss = 0
            
            for i in range(3):
                y_conv, y_pred = y_output[i * 2], y_output[i * 2 + 1]
                
                giou_loss, conf_loss, prob_loss = layer_loss(y_conv, y_pred, y_true, i)
                
                total_loss += giou_loss
                total_loss += conf_loss
                total_loss += prob_loss
            
            return total_loss
        
        return total_loss
