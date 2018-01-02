from utils import get_cnn, load, train

num_epoch = 100
name = "test-spot-gpu-instance"
X_test, _ = load(test=True)

y_pred = train(num_epoch, X_test, name, get_cnn)
