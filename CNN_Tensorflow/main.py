from training import TrainingProcedure,TestProcedure
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

LOG_DIR = '/tmp/tensorflow/mnist/logs/cnn_mnist'
TRAINING_EPOCHS = 300
DISPLAY_STEP = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 512

class Dataset:

    IMAGE_SIZE = 28
    IMAGE_CHANNELS = 1
    NUM_CLASSES = 10

    def __init__(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.X_train,self.y_train = mnist.train.images, mnist.train.labels
        self.X_test,self.y_test =  mnist.test.images, mnist.test.labels
        self.X_val,self.y_val = mnist.validation.images, mnist.validation.labels

    def train(self):
        return [self.X_train,self.y_train]

    def test(self):
        return [self.X_test,self.y_test]

    def validation(self):
        return [self.X_val,self.y_val]


def main():

    dataset = Dataset()

    train = True

    if train:
        training_procedure = TrainingProcedure(TRAINING_EPOCHS,LEARNING_RATE,BATCH_SIZE,DISPLAY_STEP,LOG_DIR)
        accuracy_stats_train,accuracy_stats_val = training_procedure.run(dataset)

        plt.plot(accuracy_stats_train,label = "TRAINING")
        plt.plot(accuracy_stats_val,label = "VALIDATION")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    test_procedure = TestProcedure(BATCH_SIZE,LOG_DIR)
    accuracy_stats_test = test_procedure.run(dataset)
    print accuracy_stats_test

if __name__ == "__main__":
    main()


