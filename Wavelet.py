import pywt
import pywt.data
import numpy as np

def decompose4ch(original, filter):
    # Wavelet transform of image, and plot approximation and details
    coeffs0 = pywt.dwt2(original[:, :, 0], filter)
    LL0, (LH0, HL0, HH0) = coeffs0
    coeffs1 = pywt.dwt2(original[:, :, 1], filter)
    LL1, (LH1, HL1, HH1) = coeffs1
    coeffs2 = pywt.dwt2(original[:, :, 2], filter)
    LL2, (LH2, HL2, HH2) = coeffs2
    coeffs3 = pywt.dwt2(original[:, :, 3], filter)
    LL3, (LH3, HL3, HH3) = coeffs3

    LL = np.stack((LL0,LL1,LL2,LL3), axis=2)
    LH = np.stack((LH0, LH1, LH2, LH3), axis=2)
    HL = np.stack((HL0, HL1, HL2, HL3), axis=2)
    HH = np.stack((HH0, HH1, HH2, HH3), axis=2)
    return (LL, LH, HL, HH)

def decompose3ch(original, filter):
    # Wavelet transform of image, and plot approximation and details
    coeffs0 = pywt.dwt2(original[:, :, 0], filter) # bior1.3
    LL0, (LH0, HL0, HH0) = coeffs0
    coeffs1 = pywt.dwt2(original[:, :, 1], filter)
    LL1, (LH1, HL1, HH1) = coeffs1
    coeffs2 = pywt.dwt2(original[:, :, 2], filter)
    LL2, (LH2, HL2, HH2) = coeffs2

    LL = np.stack((LL0,LL1,LL2),axis=2)
    LH = np.stack((LH0, LH1, LH2), axis=2)
    HL = np.stack((HL0, HL1, HL2), axis=2)
    HH = np.stack((HH0, HH1, HH2), axis=2)
    return (LL, LH, HL, HH)

def combine3ch(LL, LH, HL, HH, filter):
    LL0 = LL[0, :, :, 0]
    LL1 = LL[0, :, :, 1]
    LL2 = LL[0, :, :, 2]

    LH0 = LH[0, :, :, 0]
    LH1 = LH[0, :, :, 1]
    LH2 = LH[0, :, :, 2]

    HL0 = HL[0, :, :, 0]
    HL1 = HL[0, :, :, 1]
    HL2 = HL[0, :, :, 2]

    HH0 = HH[0, :, :, 0]
    HH1 = HH[0, :, :, 1]
    HH2 = HH[0, :, :, 2]
    coeffs0 = (LL0, (LH0, HL0, HH0))
    coeffs1 = (LL1, (LH1, HL1, HH1))
    coeffs2 = (LL2, (LH2, HL2, HH2))

    R = pywt.idwt2(coeffs0, filter)
    G = pywt.idwt2(coeffs1, filter)
    B = pywt.idwt2(coeffs2, filter)

    return np.stack((R, G, B), axis=2)
