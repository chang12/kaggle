from utils import get_cnn, train

num_epoch = 2
name = "test AWS GPU EC2 instance"

train(num_epoch, name, get_cnn, batch_size=32, learning_rate=0.05, save_every=2)
