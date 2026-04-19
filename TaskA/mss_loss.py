import numpy as np
import librosa

def multi_scale_spectral_loss(target, candidate, stft_configs=None):
    """
    Multiresolution spectral loss using multiple STFT configurations.
    
    Args:
        target: Target audio signal
        candidate: Candidate audio signal  
        sample_rate: Sample rate (not currently used but kept for future extensions)
        stft_configs: List of tuples (n_fft, hop_length). If None, uses default configurations.
    """
    if stft_configs is None:
        stft_configs = [(512, 128), (2048, 512), (8192, 2048)]
    
    # Compute STFTs at multiple resolutions (log-magnitude L1: the
    # unnormalized displacement IRs have amplitudes ~1e-8, so a linear-
    # magnitude loss compresses the dynamic range and is uninformative
    # across candidates).
    eps = 1e-10
    losses = []
    for n_fft, hop_length in stft_configs:
        target_stft = np.abs(librosa.stft(target, n_fft=n_fft, hop_length=hop_length))
        candidate_stft = np.abs(librosa.stft(candidate, n_fft=n_fft, hop_length=hop_length))
        loss = np.mean(np.abs(np.log(target_stft + eps) - np.log(candidate_stft + eps)))
        losses.append(loss)
    return np.mean(losses)