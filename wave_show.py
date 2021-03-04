import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as nf
import scipy.io.wavfile as wavf
import os

root = "/home/feng/workspace/pycharm/VOICE/demo_data/origin_data"

if __name__ == "__main__":
    for filename in os.listdir(root):
        sr, data_ori = wavf.read((os.path.join(root, filename)))
        x = [x / 12 for x in range(len(data_ori))]
        comp_arr = nf.fft(data_ori)
        # 滤波
        freqs = nf.fftfreq(data_ori.size, x[1] - x[0])
        data_index = np.argwhere(freqs < 0.3)
        comp_arr[data_index] = 0
        # 傅里叶逆变换
        data_ifft = nf.ifft(comp_arr).real
        signal = (data_ifft / np.max(np.abs(data_ifft)) * 2147483647).astype(np.int32)
        wavf.write("/home/feng/workspace/pycharm/VOICE/demo_data/filter_data/" + filename, sr, data_ori)
        wavf.write("/home/feng/workspace/pycharm/VOICE/demo_data/filter_data/" + "filter_" + filename, sr, signal)

        # show:原始图像,滤波后图像，频谱
        plt.figure('FFT', facecolor='lightgray')
        plt.subplot(121)
        plt.title('Time Domain', fontsize=16)
        plt.grid(linestyle=':')
        plt.plot(x, data_ori, label=r'$y$')
        plt.plot(x, data_ifft, color='orangered', linewidth=5, alpha=0.5, label=r'$y$')
        plt.subplot(122)
        pows = np.abs(comp_arr)
        plt.title('Frequency Domain', fontsize=16)
        plt.grid(linestyle=':')
        plt.plot(freqs[freqs > 0], pows[freqs > 0], color='orangered', label='frequency')

        plt.legend()
        plt.savefig('fft.png')
        plt.show()
