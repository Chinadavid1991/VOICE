import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from data_tools import scaled_in
from model_cnet2 import cnet


def training(path_save_spectrogram, weights_path, name_model, training_from_scratch, epochs, batch_size, nf):
    """ This function will read noisy voice and clean voice spectrograms created by data_creation mode,
    and train a Unet model on this dataset for epochs and batch_size specified. It saves best models to disk regularly
    If training_from_scratch is set to True it will train from scratch, if set to False, it will train
    from weights (name_model) provided in weights_path
    """
    # load noisy voice & clean voice spectrograms created by data_creation mode
    # X_in = np.load(path_save_spectrogram +'noisy_voice_amp_db'+".npy")
    X_in1 = np.load(path_save_spectrogram + 'noise_amp_db' + str(nf) + ".npy")
    # X_in2 = np.load(path_save_spectrogram +'noisy_voice_amp_db'+".npy")
    X_in2 = np.load(path_save_spectrogram + 'voice_amp_db' + str(nf) + ".npy")
    # Model of noise to predict

    r = 0
    X_in1 = X_in1[:, :, :]
    X_in2 = X_in2[:, :, :] * (1 - r) + X_in1 * r
    c = np.mean(np.abs(X_in2))
    print(c)

    negn = X_in1.shape[0]
    posn = X_in2.shape[0]

    Z_ou = np.array([0] * negn + [1] * posn)

    X_in = np.concatenate((X_in1, X_in2), axis=0)
    nscales = 5
    for i in range(1, nscales):
        X_in = np.concatenate((X_in, X_in1 * (i / nscales), X_in2 * (i / nscales)), axis=0)
        Z_ou = np.concatenate((Z_ou, np.array([0] * negn + [1] * posn)), axis=0)

    X_in = scaled_in(X_in)

    # Check shape of spectrograms
    print(X_in.shape)

    X_in = X_in[:, :, :]
    X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2], 1)

    print(X_in.shape)

    X_train, X_test, z_train, z_test = train_test_split(X_in, Z_ou, test_size=0.10, shuffle=True)

    # If training from scratch
    if training_from_scratch:
        nn = cnet()
    else:
        nn = cnet(pretrained_weights=weights_path + name_model + '.h5')

    # Save best models to disk during training
    checkpoint = ModelCheckpoint(weights_path + '/model_save.h5', verbose=1, monitor='val_loss', save_best_only=True,
                                 mode='auto')

    nn.summary()
    time.sleep(2)

    INIT_LR = 1e-5
    losses = {
        "disc_output": "MeanAbsoluteError",
    }

    lossWeights = {"disc_output": 1.0}
    opt = Adam(lr=INIT_LR, decay=INIT_LR / epochs)
    nn.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])
    # Training
    validation_data = (X_test, {
        "disc_output": z_test})

    history = nn.fit(x=X_train, y={
        "disc_output": z_train}, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[checkpoint],
                     verbose=1, validation_data=validation_data)

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, label='Training class loss')
    plt.plot(epochs, val_loss, label='Validation class loss')

    plt.yscale('log')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
