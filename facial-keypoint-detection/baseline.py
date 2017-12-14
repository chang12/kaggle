from datetime import datetime

import numpy as np
from pandas.io.parsers import read_csv

from utils import prepare_submission_slave


test_data_shape = read_csv("/Users/chang12/projects/kaggle/facial-keypoint-detection/test.csv").values.shape
df = read_csv("/Users/chang12/projects/kaggle/facial-keypoint-detection/training.csv")
df = df.dropna()
y = df[df.columns[:-1]].values
y_avg = np.reshape(np.average(y, axis=0), [1, 30])
y_baseline = np.repeat(y_avg, test_data_shape[0], axis=0)

datetime_now = datetime.now().strftime("%Y%m%d_%H:%M:%S")
prepare_submission_slave(y_baseline, "submissions/{}_baseline.csv".format(datetime_now))
