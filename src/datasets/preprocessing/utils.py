import numpy as np

def convert_pixel_to_hu(dcm):
    if not hasattr(dcm, 'pixel_array'):
        raise ValueError('DICOM file does not contain pixel data.')
    pixel_array = dcm.pixel_array.astype(np.float32)
    slope = getattr(dcm, 'RescaleSlope', 1)
    intercept = getattr(dcm, 'RescaleIntercept', 0)
    return slope * pixel_array + intercept

def apply_window(img, window_center, window_width):
    if window_center is None or window_width is None:
        raise ValueError('Missing window_center or window_width parameters.')
    window_width = max(window_width, 1.0)
    min_val = window_center - (window_width / 2)
    max_val = window_center + (window_width / 2)
    img = np.clip(img, min_val, max_val)
    img = ((img - min_val) / (max_val - min_val))
    return img.astype(np.float32)