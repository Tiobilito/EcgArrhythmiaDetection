from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Flatten, Dense
from keras.optimizers import Adam
from config import LABELS, HYPERPARAMS

def build_cnn(input_shape, filters, dropout_rates):
    model = Sequential()
    for f, dr in zip(filters, dropout_rates):
        if model.layers:
            model.add(Conv1D(f, 5, activation='relu'))
        else:
            model.add(Conv1D(f, 5, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Dropout(dr))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(LABELS), activation='softmax'))

    opt = Adam(learning_rate=HYPERPARAMS['learning_rate'])
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
