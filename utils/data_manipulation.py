import torch
import numpy as np
from scipy.signal import find_peaks, hilbert
import pywt

import matplotlib.pyplot as plt

class FeatureEngineering:
    def __init__(self):
        return   
    
    def min_max_spectra(self, spectra, NORM=True):
        """
        Create context for spectra by extracting minimum and maximum values for each spectrum.

        Parameters:
        spectra (numpy.ndarray): Array of shape (x, y) containing x spectra with y bins.
        NORM (bool, default=True): Flag to indicate if the spectra should be normalized

        Returns:
        X (torch.Tensor): Tensor of shape (x, y+2) containing the (normalized) spectra 
                            combined with the engineered columns
        """
        context = np.zeros((spectra.shape[0], 2))
        
        context[:, 0] = np.min(spectra, axis=1)
        context[:, 1] = np.max(spectra, axis=1)
        
        if NORM:
            # L2 normalization
            spectra = torch.from_numpy(np.array([spectrum / np.linalg.norm(spectrum, keepdims=True) for spectrum in spectra])).float()
        else:
            spectra = torch.tensor(spectra).float()
        context = torch.tensor(context).float()
        # combine data
        X = torch.cat([spectra,context],1)
        return X


def subtract_continuum(spectra, continuum=None):
    """
    Returns continuum-subtracted spectra

    Parameters:
    spectra (numpy.ndarray): Array of shape (x, y) containing x spectra with y bins.
    continuum (numpy.ndarray, default=None): Array of shape (x, y) containing the estimated continuum for each spectrum.

    Returns:
    numpy.ndarray: Array of shape (x, y) containing the continuum-subtracted spectra.
    """
    if continuum is not None:        
        return spectra - continuum
    else:
        continuum = _estimate_continuum_multiple(spectra)
        return spectra-continuum

def _estimate_continuum_multiple(X, polyorder=3, peaks_prominence=0.5):
    """
    Estimate continuum for multiple spectra stored in a numpy array.
    
    Parameters:
    X (numpy.ndarray): Array of shape (x, y) containing x spectra with y bins.
    polyorder (int): Order of the polynomial for fitting the continuum.
    peaks_prominence (float): Prominence parameter for peak detection.
    
    Returns:
    numpy.ndarray: Array of shape (x, y) containing the estimated continuum for each spectrum.
    """
    num_spectra, spectrum_length = X.shape
    continuum_all = np.zeros_like(X)
    x = np.arange(spectrum_length)
    
    for i in range(num_spectra):
        y = X[i]
        
        # Find peaks to exclude from continuum estimation
        peaks, _ = find_peaks(y, prominence=peaks_prominence)
        
        # Create a mask to exclude peaks
        mask = np.ones(spectrum_length, dtype=bool)
        mask[peaks] = False
        
        # Fit a polynomial to the non-peak regions
        coeffs = np.polyfit(x[mask], y[mask], polyorder)
        continuum = np.polyval(coeffs, x)
        
        continuum_all[i] = continuum
    
    return continuum_all

def hilbert_transform(spectra):
    """
    Apply Hilbert transform to the given spectra.

    Parameters:
    spectra (numpy.ndarray): The input spectra to be transformed.

    Returns:
    numpy.ndarray: The Hilbert transformed spectra.
    """
    analytic_signal = hilbert(spectra)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi)
    return np.hstack([amplitude_envelope, instantaneous_frequency])

def wavelet_transform(spectra, wavelet='db4', level=3):
    """
    Apply wavelet transform to the given spectra.

    Parameters:
    spectra (numpy.ndarray): The input spectra to be transformed.
    wavelet (str, default='db4'): The wavelet function to be used for the transformation.
    level (int, default=3): The number of decomposition levels.

    Returns:
    numpy.ndarray: The wavelet transformed spectra.
    """
    coeffs = pywt.wavedec(spectra, wavelet, level=level)
    return np.concatenate([pywt.upcoef('a', c, wavelet, level=i+1, take=len(spectra)) 
                           for i, c in enumerate(coeffs)])