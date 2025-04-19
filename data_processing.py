import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from config import TRAIN_FILE, TEST_FILE, HYPERPARAMS

def load_data():
    train_df = pd.read_csv(TRAIN_FILE, header=None)
    test_df  = pd.read_csv(TEST_FILE,  header=None)
    return train_df, test_df

def preprocess(df):
    X = df.iloc[:, :187].astype(np.float32)
    y = df.iloc[:, 187].astype(int)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler

def resample_data(X, y):
    ros = RandomOverSampler(random_state=HYPERPARAMS['random_state'])
    return ros.fit_resample(X, y)

def prepare_datasets():
    train_df, test_df = load_data()
    X_train_raw, y_train_raw, scaler = preprocess(train_df)
    X_test_raw,  y_test_raw,  _      = preprocess(test_df)

    if HYPERPARAMS['oversample']:
        X_train_raw, y_train_raw = resample_data(X_train_raw, y_train_raw)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_raw, y_train_raw,
        test_size=HYPERPARAMS['test_size'],
        random_state=HYPERPARAMS['random_state'],
        stratify=y_train_raw
    )

    # reshape para modelos 1D
    X_train = X_train.reshape(-1, 187, 1)
    X_val   = X_val.reshape(-1, 187, 1)
    X_test  = X_test_raw.reshape(-1, 187, 1)

    import tensorflow as tf
    y_train_enc = tf.keras.utils.to_categorical(y_train)
    y_val_enc   = tf.keras.utils.to_categorical(y_val)
    y_test_enc  = tf.keras.utils.to_categorical(y_test_raw)

    return (X_train, y_train_enc), (X_val, y_val_enc), (X_test, y_test_enc), scaler
