from utils import get_cnn, train

num_epoch = 500
name = "cover_nan_data_500_epochs"

y_pred = train(num_epoch, name, get_cnn)
