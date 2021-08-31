import cv2
import numpy as np
from .image import transform

def limit_roi(roi, im_height, im_width):
    """limit roi to image border"""
    left = max(0, roi[0])
    top = max(0, roi[1])
    right = min(im_width - 1, roi[2])
    bottom = min(im_height - 1, roi[3])

    return [left, top, right, bottom]
    
def crop_image(im, roi, roi_shape):
    """Crop im patches in roi to roi_shape, for roi exceed the image border,
    the exceeded part will be padded with CONSTANT
    
    Parameters
    ----------
    im: numpy.ndarray.uint8
    roi: list
        image roi to crop with
    roi_shape: list
        image shape to crop to
        
    Returns:
    -------
    im_roi: numpy.float32
        cropped image roi
    """
    roi = [int(v) for v in roi]
    cut_roi = limit_roi(roi, im.shape[0], im.shape[1])

    if len(im.shape) == 3:
        im_roi = im[cut_roi[1]:cut_roi[3]+1, cut_roi[0]:cut_roi[2]+1, :]
    else:
        im_roi = im[cut_roi[1]:cut_roi[3]+1, cut_roi[0]:cut_roi[2]+1]
    im_roi = cv2.copyMakeBorder(im_roi,
                                cut_roi[1] - roi[1], roi[3] - cut_roi[3],
                                cut_roi[0] - roi[0], roi[2] - cut_roi[2],
                                cv2.BORDER_CONSTANT)
    roi_shape = (roi_shape[-1], roi_shape[-2])

    im_roi = cv2.resize(im_roi, roi_shape, interpolation=cv2.INTER_LINEAR)
         
    im_roi = im_roi.astype(np.float32)
    return im_roi

def get_roi_images(im, roi_list, roi_shape, input_mean, scale):
    """get roi image data for cnn input
    
    Parameters:
    ----------
    
    im: numpy.uint8
    roi_list: list(list()) or list
        list of roi lists or one roi list
    roi_shape: list
        [roi_height, roi_width]
    input_meat: list
        [r, g, b]
    
    Returns:
    -------
    roi_images: numpy.float32
        [n, h, w, c]
    """
    
    if isinstance(roi_list[0], list):
        roi_images = [crop_image(im, roi, roi_shape) for roi in roi_list]
        roi_images = np.vstack([transform(roi_im, input_mean, scale=scale) for roi_im in roi_images])   
    else:
        roi_images = crop_image(im, roi_list, roi_shape)
        roi_images = transform(roi_images, input_mean, scale=scale)

    return roi_images