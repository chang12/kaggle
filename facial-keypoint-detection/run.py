from utils import get_cnn, train

num_epoch = 100
name = "batch_size=32&learning_rate=0.05&save in every 20 epochs"

y_pred = train(num_epoch, name, get_cnn, batch_size=32, learning_rate=0.05, save_every=20)
