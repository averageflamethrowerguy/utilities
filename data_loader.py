import sys
sys.path.append('../')

from utilities import Dataset
import torch
import pandas as pd
import numpy as np

def load_csv(csv, LOOKBACK_DISTANCE, PREDICTION_RANGE, BATCH_SIZE, dtype):
    # read the CSV
    df = pd.read_csv(csv, parse_dates=False, na_values=0)
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    print("Dataframe length: " + str(len(df)))

    # function to get the number of minutes from the open
    def get_minutes_from_open(time):
        hour_min_sec = time.split(' ')[1]
        minutes = int(hour_min_sec.split(':')[1]) + 60*int(hour_min_sec.split(':')[0]) - 570
        return minutes / 195 - 1

    df['time'] = df['time'].apply(get_minutes_from_open)
    df.drop(df[df['time'] > 1].index, inplace=True)
    df.drop(df[df['time'] < -1].index, inplace=True)
    df.fillna(0)
    print("Dataframe length adjusted: " + str(len(df)))

    df['1_min_increase'] = df.open.pct_change()
    df['1_min_volume_increase'] = df.volume.pct_change()

    # delete the first 1 row
    df = df.iloc[1:]
    df.fillna(0)

    df = df.drop(labels=['open', 'high', 'low', 'close', 'volume'], axis=1)

    NUM_FEATURES=len(df.columns)

    print('Number of features: ' + str(NUM_FEATURES))

    yval_tensor = torch.tensor(df['1_min_increase'][LOOKBACK_DISTANCE + PREDICTION_RANGE:].values)
    print('Length of yvals: ' + str(len(yval_tensor)))

    # Splits into train and test data
    train_set_size = int(len(yval_tensor) * 0.8)
    yval_train = yval_tensor[:train_set_size]
    yval_test = yval_tensor[train_set_size:]

    xval_tensor = torch.tensor(df.values)
    xval_tensor[np.isnan(xval_tensor)] = 0  # takes care of 8 nan values that slipped through
    xval_train = xval_tensor[:train_set_size + LOOKBACK_DISTANCE]
    xval_test = xval_tensor[train_set_size:]

    train_set = Dataset.Dataset(
        yval_train.cuda().to(torch.float32),
        xval_train.cuda().to(dtype),
        LOOKBACK_DISTANCE
    )

    test_set = Dataset.Dataset(
        yval_test.cuda(),
        xval_test.cuda().to(dtype),
        LOOKBACK_DISTANCE
    )

    params = {'batch_size': BATCH_SIZE, 'shuffle': False}

    train_generator = torch.utils.data.DataLoader(train_set, **params)
    test_generator = torch.utils.data.DataLoader(test_set, **params)

    return train_generator, test_generator, NUM_FEATURES
