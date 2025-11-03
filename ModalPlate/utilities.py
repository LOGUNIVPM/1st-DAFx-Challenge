# ---------------------------------------------------------
import sys
import os
import time

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# Import logger to ensure global print override is active
try:
    import logger
except ImportError:
    pass  # Logger module may not be available in all contexts

PRINT_PROGRESS = False

# ---------------------------------------------------------
# Source: https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=50):

    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """

    if not PRINT_PROGRESS:
        return

    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration >= total:
        sys.stdout.write('\r\n')
    sys.stdout.flush()


# ---------------------------------------------------------
def soundsc(audio_data, sample_rate, db_peak=-3.0, should_block=True):
    """

    :param audio_data:
    :param sample_rate:
    :return:
    """
    gain = (10 ** (db_peak / 20.0)) / np.max(np.abs(audio_data))
    sd.play(audio_data * gain, sample_rate)
    audio_data[0:50] *= np.atleast_2d((np.r_[0:50] / 50.0)).T
    audio_data[-50:] *= np.atleast_2d((np.r_[50:0:-1]-1)/50.0).T
    
    if should_block:
        time.sleep((float(len(audio_data)) / sample_rate) + 0.5)


def sfwrite(filename, audio_data, sample_rate):
    """

    :param filename:
    :param audio_data:
    :param sample_rate:
    :return:
    """
    if not os.path.isdir("audio_output"):
        os.makedirs("audio_output")

    write(filename, int(sample_rate), audio_data / np.abs(audio_data).max())
    print(f"audio written to {os.path.abspath('./audio_output')}")