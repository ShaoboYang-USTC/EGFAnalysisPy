import sys
sys.path.append('DisperPicker')
from train_cnn import CNN
from config import Config

config = Config()
cnn = CNN()
cnn.train(passes = config.training_step)