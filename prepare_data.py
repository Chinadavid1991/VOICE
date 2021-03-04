import os
import librosa
import soundfile as sf
from data_tools import audio_files_to_numpy
from data_tools import blend_noise_randomly, noise_mixrandomly, numpy_audio_to_matrix_spectrogram
import numpy as np


def create_data(noise_dir, voice_dir, path_save_time_serie, path_save_sound, path_save_spectrogram, sample_rate,
                min_duration, frame_length, hop_length_frame, hop_length_frame_noise, nb_samples, n_fft, hop_length_fft,
                mode):
    """This function will randomly blend some clean voices from voice_dir with some noises from noise_dir
    and save the spectrograms of noisy voice, noise and clean voices to disk as well as complex phase,
    time series and sounds. This aims at preparing datasets for denoising training. It takes as inputs
    parameters defined in args module"""

    alllist_noise_files = os.listdir(noise_dir)
    alllist_voice_files = os.listdir(voice_dir)

    def remove_ds_store(lst):
        """remove mac specific file if present"""
        if '.DS_Store' in lst:
            lst.remove('.DS_Store')

        return lst

    list_noise_files = remove_ds_store(alllist_noise_files)
    list_voice_files = remove_ds_store(alllist_voice_files)

    nb_voice_files = len(alllist_voice_files)
    nb_noise_files = len(alllist_noise_files)

    n = 1
    list_list_voice_id = list()
    for k in range(0, n):
        list_voice_id = list()
        for id in range(nb_voice_files):
            if (k / n * nb_voice_files <= id) & (id <= ((k + n) / n) * nb_voice_files):
                list_voice_id.append(id)
        list_list_voice_id.append(list_voice_id)

    list_list_noise_id = list()
    for k in range(0, n):
        list_noise_id = list()
        for id in range(nb_noise_files):
            if (k / n * nb_noise_files <= id) & (id <= ((k + n) / n) * nb_noise_files):
                list_noise_id.append(id)
        list_list_noise_id.append(list_noise_id)

    ii = 0
    for list_noise_id in list_list_noise_id:
        list_noise_files = []
        for id in list_noise_id:
            list_noise_files.append(alllist_noise_files[id])

        for list_voice_id in list_list_voice_id:
            list_voice_files = []
            for id in list_voice_id:
                list_voice_files.append(alllist_voice_files[id])

            # Extracting noise and voice from folder and convert to numpy
            noise = audio_files_to_numpy(noise_dir, list_noise_files, sample_rate,
                                         frame_length, hop_length_frame_noise * 1, min_duration, mode)

            voice = audio_files_to_numpy(voice_dir, list_voice_files,
                                         sample_rate, frame_length, hop_length_frame * 10, min_duration, mode)

            # Blend some clean voices with random selected noises (and a random level of noise)

            prod_voice, prod_noise, prod_noisy_voice = blend_noise_randomly(
                voice, noise, nb_samples, frame_length, 1)

            # To save the long audio generated to disk to QC:
            noisy_voice_long = prod_noisy_voice.reshape(1, nb_samples * frame_length)
            sf.write(path_save_sound + '/noisy_voice_long' + str(ii) + '.wav', noisy_voice_long[0, :], sample_rate)
            voice_long = prod_voice.reshape(1, nb_samples * frame_length)
            sf.write(path_save_sound + '/voice_long' + str(ii) + '.wav', voice_long[0, :], sample_rate)
            noise_long = prod_noise.reshape(1, nb_samples * frame_length)
            sf.write(path_save_sound + '/noise_long' + str(ii) + '.wav', noise_long[0, :], sample_rate)

            # Squared spectrogram dimensions
            dim_square_spec = int(n_fft / 2) + 1

            # Create Amplitude and phase of the sounds
            m_amp_db_voice, m_pha_voice = numpy_audio_to_matrix_spectrogram(
                prod_voice, dim_square_spec, n_fft, hop_length_fft)
            m_amp_db_noise, m_pha_noise = numpy_audio_to_matrix_spectrogram(
                prod_noise, dim_square_spec, n_fft, hop_length_fft)
            m_amp_db_noisy_voice, m_pha_noisy_voice = numpy_audio_to_matrix_spectrogram(
                prod_noisy_voice, dim_square_spec, n_fft, hop_length_fft)

            # Save to disk for Training / QC
            # np.save(path_save_time_serie + '/voice_timeserie'+str(ii), prod_voice)
            # np.save(path_save_time_serie + '/noise_timeserie'+str(ii), prod_noise)
            # np.save(path_save_time_serie + '/noisy_voice_timeserie'+str(ii), prod_noisy_voice)

            np.save(path_save_spectrogram + '/voice_amp_db' + str(ii), m_amp_db_voice)
            np.save(path_save_spectrogram + '/noise_amp_db' + str(ii), m_amp_db_noise)
            # np.save(path_save_spectrogram + '/noisy_voice_amp_db'+str(ii), m_amp_db_noisy_voice)

            np.save(path_save_spectrogram + '/voice_pha_db' + str(ii), m_pha_voice)
            np.save(path_save_spectrogram + '/noise_pha_db' + str(ii), m_pha_noise)
            # np.save(path_save_spectrogram + '/noisy_voice_pha_db'+str(ii), m_pha_noisy_voice)

            ii = ii + 1
