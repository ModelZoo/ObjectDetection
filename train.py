from model_zoo.trainer import BaseTrainer
from model_zoo import flags
import numpy as np

flags.define('epochs', 10000)
flags.define('model_class_name', 'YOLOV3Model')
flags.define('number_anchors', 3)
flags.define('number_classes', 1)


class Trainer(BaseTrainer):
    """
    Train Image Classification Model.
    """
    
    def data(self):
        """
        Prepare fashion mnist data.
        :return:
        """
        inputs = np.zeros(shape=[10, 416, 416, 3])
        outputs = np.zeros(shape=[10, 13, 13, 1024])
        return (inputs, outputs), None


if __name__ == '__main__':
    Trainer().run()
