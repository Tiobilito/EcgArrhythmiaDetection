from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from config import LABELS, HYPERPARAMS

def build_rnn(input_shape, units, dropout_rates):
    model = Sequential()
    # Capas LSTM configurables
    for i, (u, dr) in enumerate(zip(units, dropout_rates)):
        return_seq = (i < len(units)-1)
        if i == 0:
            model.add(LSTM(u, return_sequences=return_seq, input_shape=input_shape))
        else:
            model.add(LSTM(u, return_sequences=return_seq))
        model.add(Dropout(dr))

    model.add(Dense(len(LABELS), activation='softmax'))

    opt = Adam(learning_rate=HYPERPARAMS['learning_rate'])
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
