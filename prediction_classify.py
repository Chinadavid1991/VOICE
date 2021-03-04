import numpy as np
from tensorflow.keras.models import load_model
from data_tools import audio_file_to_numpy, numpy_audio_to_matrix_spectrogram
from data_tools import scaled_in


def prediction(weights_path, name_model, audio_dir_prediction, audio_input_prediction
               , sample_rate, min_duration, frame_length, hop_length_frame, n_fft,
               hop_length_fft, mode):
    """ This function takes as input pretrained weights, noisy voice sound to denoise, predict
    the denoise sound and save it to disk.
    """
    loaded_model = load_model(weights_path + '/' + name_model + '.h5')
    print("Loaded model from disk")

    # Extracting noise and voice from folder and convert to numpy
    audio = audio_file_to_numpy(audio_dir_prediction, str(audio_input_prediction), sample_rate,
                                frame_length, hop_length_frame, min_duration, mode)

    # Dimensions of squared spectrogram
    dim_square_spec = int(n_fft / 2) + 1
    print("dim_square_spec:{}".format(dim_square_spec))

    m_amp_db_audio, m_pha_audio = numpy_audio_to_matrix_spectrogram(
        audio, dim_square_spec, n_fft, hop_length_fft)

    # global scaling to have distribution -1/1
    X_in = scaled_in(m_amp_db_audio)
    # Reshape for prediction
    X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2], 1)
    # Prediction using loaded network
    X_pred = loaded_model.predict(X_in)
    audio_class = audio
    for i in range(audio_class.shape[0]):
        audio_class[i, :] = audio_class[i, :] * 0 + X_pred[i]

    # Number of frames
    nb_samples = audio_class.shape[0]
    # Save all frames in one file
    audio_class_long = audio_class.reshape(1, nb_samples * frame_length)
    # librosa.output.write_wav(dir_save_prediction + audio_output_prediction, denoise_long[0, :], sample_rate)
    # sf.write(dir_save_prediction + audio_output_prediction, res, sample_rate)

    print("声音的强度{:.2f}".format(np.mean(np.abs(m_amp_db_audio))))
    print("漏水的概率{:.2f}".format(np.mean(audio_class_long)))
