from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten


def cnet(input_size=(128, 128, 1)):
    # size filter input
    size_filter_in = 128*128

    model = Sequential(
        [
            Input(shape=input_size),
            Flatten(),
            Dense(size_filter_in // 64, activation="relu", use_bias=True),
            Dense(size_filter_in // 512, activation="relu", use_bias=True),
            Dropout(0.5),
            Dense(1, activation="sigmoid", name="disc_output"),
        ]
    )

    return model
