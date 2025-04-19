from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from config import LABELS, HYPERPARAMS

def build_ann(input_shape, hidden_sizes, dropout_rates):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))

    # Capas densas configurables
    for nh, dr in zip(hidden_sizes, dropout_rates):
        model.add(Dense(nh, activation='relu'))
        model.add(Dropout(dr))

    model.add(Dense(len(LABELS), activation='softmax'))

    opt = Adam(learning_rate=HYPERPARAMS['learning_rate'])
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
