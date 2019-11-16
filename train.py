from model_zoo.trainer import BaseTrainer
from model_zoo import flags
import numpy as np

flags.DEFINE_integer('epochs', 10000)
flags.DEFINE_string('model_class_name', 'YOLOV3Model')


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
