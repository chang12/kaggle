from utils import get_cnn, train

num_epoch = 500
name = "batch_size=32&learning_rate=0.05"

y_pred = train(num_epoch, name, get_cnn, batch_size=32, learning_rate=0.05)
