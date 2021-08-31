import numpy as np
import cv2
import math


def resize(im, target_size, max_size, stride=0, interpolation=cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale


def transform(im, pixel_means, scale=1.0):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    if len(im.shape) == 3 and im.shape[2] == 3:
        im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]), dtype=np.float32)
        for i in range(3):
            im_tensor[0, i, :, :] = (im[:, :, 2 - i] - pixel_means[2 - i]) * scale
    elif len(im.shape) == 2:
        im_tensor = np.zeros((1, 1, im.shape[0], im.shape[1]), dtype=np.float32)
        im_tensor[0, 0, :, :] = (im[:, :] - pixel_means[0]) * scale
    else:
        raise ValueError("can not transform image successfully")
    return im_tensor


def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im


def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
    else:
        raise Exception('Sorry, unimplemented.')
    return all_tensor

    
def addMaskImage(img):
    """
    add random mask on image by some pure image patch 
    to be improved
    :param im: BGR image input by opencv
    """
    [h, w, c] = img.shape
    h_start = np.random.randint(h/2,h-1)
    w_start = np.random.randint(w/2, w-1)
    img[h_start:h-1, :,0]= np.random.randint(0,120)
    img[h_start:h-1, :,1]= np.random.randint(0,120)    
    img[h_start:h-1, :,2]= np.random.randint(0,120)  
    img[:,w_start:w-1,0]= np.random.randint(0,120)
    img[:,w_start:w-1,1]= np.random.randint(0,120)    
    img[:,w_start:w-1,2]= np.random.randint(0,120)    
    img = np.uint8(img)
    return img, h_start, w_start


def Contrast(img):
    """
    adjust the contrast of the image,  w.r.t
    http://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
    :param im: BGR image input by opencv
    :param factor: constract factor, [-128, 128]
    :method: fvalue = 259.0/255.0 * (factor + 255.0)/(259.0-factor)
             pixel = (pixel - 128.0) * fvalue + 128.0
    :return modified image
    """
    factor = 2 * (np.random.rand() - 0.5) * 128
    assert (factor <= 128 and factor >= -128), 'contract factor value wrong'
    fvalue = 259.0/255.0 * (factor + 255.0)/(259.0-factor)
    img = np.round((img - 128.0)*fvalue + 128.0)
    img = np.where(img > 255, 255, img)
    img = np.where(img < 0, 0, img)
    img = np.uint8(img)
    return img

def Bright(img):
    """
    adjust the brightneess of the image
    :param im: BGR image input by opencv
    :param factor: brightness factor, [0, 2], <1 dark, >1 bright
    :method:
    :return modified image
    """
    factor = 2 * np.random.rand()
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    V= V* np.float(factor)
    V = np.where(V>255, 255,V)
    V = np.where(V<0, 0, V)
    HSV[:,:,2] = np.uint8(V)
    BGR = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
    return BGR

def Saturation(img):
    """
    adjust the saturation of the image
    :param im: BGR image input by opencv
    :param factor: brightness factor, [0, 2]
    :method:
    :return modified image
    """
    factor = 2 * np.random.rand()
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    S= S* np.float(factor)
    S = np.where( S>255, 255,S)
    S = np.where( S<0, 0, S)
    HSV[:,:,1] = np.uint8(S)
    BGR = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
    return BGR

def GuassBlur(img, kernel):
    """
    add gaussian blur
    :param im: BGR image input by opencv
    :param kernel: 3-10
    :method:
    :return modified image
    """
    img = cv2.blur(img, (kernel, kernel))
    return img

def BoxBlur(img, kernel):
    """
    add box blur
    """
    kernel = np.ones((kernel, kernel), np.float32)/(kernel*kernel)
    dst = cv2.filter2D(img, -1, kernel)
    return dst

def MedianBlur(img, odd_kernel):
    """
    add median blur
    """
    dst = cv2.medianBlur(img, odd_kernel)
    return dst

def BilateralBlur(img, diameter):
    """
    rft http://people.csail.mit.edu/sparis/bf_course/
    add bilateral blur
    """
    dst = cv2.bilateralFilter(img,diameter,75,75)
    return dst

def MotionBlur(img, kernel):
    """
    add mothon blur
    """
    kernel_motion_blur = np.zeros((kernel, kernel))
    kernel_motion_blur[int((kernel-1)/2), :] = np.ones(kernel)
    kernel_motion_blur = kernel_motion_blur / kernel
    dst = cv2.filter2D(img, -1, kernel_motion_blur)
    return dst

def CreateMotionKernel(kernel):
    """
    create a random trajectory kernel for motion blur
    """
    TrajSize = 64
    anxiety = 0.2* np.random.rand()
    numT = 10
    MaxTotalLength =10
    TotLength = 0
    #term determining, at each sample, the strengh of the component leating towards the previous position
    centripetal = 0.7 *  np.random.rand()
    #term determining, at each sample, the random component of the new direction
    gaussianTerm =10 * np.random.rand()
    #probability of having a big shake, e.g. due to pressing camera button or abrupt hand movements
    freqBigShakes = 3 *np.random.rand()
    #v is the initial velocity vector, initialized at random direction
    init_angle = 360 * np.random.rand()
    #initial velocity vector having norm 1
    v0 = math.cos(init_angle / 180.0 * math.pi) + 1.0j * math.sin(init_angle/ 180.0 * math.pi)
    #the speed of the initial velocity vector
    v = v0* MaxTotalLength/(numT-1);

    if anxiety > 0:
        v = v0 * anxiety
    # initialize the trajectory vector
    x = np.zeros(numT,dtype = np.complex);

    abruptShakesCounter = 0
    for t in range(numT-1):
        # determine if there is an abrupt (impulsive) shake
        if np.random.rand() < freqBigShakes * anxiety:
            #if yes, determine the next direction which is likely to be opposite to the previous one
            nextDirection = 2 * v * (np.exp( 1.0j * (math.pi + (np.random.rand() - 0.5))))
            abruptShakesCounter = abruptShakesCounter + 1
        else:
            nextDirection=0

        #determine the random component motion vector at the next step
        dv = nextDirection + anxiety * (gaussianTerm * (np.random.randn()- + 1.0j * np.random.randn()) - centripetal * x[t]) * (MaxTotalLength / (numT - 1))
        v = v + dv
        # velocity vector normalization
        v = (v / np.abs(v)) * MaxTotalLength / (numT - 1)
        #print v
        x[t + 1] = x[t] + v
        # compute total length
        #TotLength=TotLength+np.abs(x([t+1]-x[t]))
    x_real = []
    x_imag = []
    for elem in x:
        x_real.append(elem.real)
        x_imag.append(elem.imag)
    x_real = np.round((x_real - np.min(x_real))/(np.max(x_real) - np.min(x_real)) * kernel-0.5)
    x_imag = np.round((x_imag - np.min(x_imag))/(np.max(x_imag) - np.min(x_imag)) * kernel-0.5)
    for idx in range(len(x_real)):
        if x_real[idx] < 0:
            x_real[idx] = 0
        if x_imag[idx] < 0:
            x_imag[idx] = 0
        if x_real[idx] > kernel -1:
            x_real[idx] =  kernel -1
        if x_imag[idx]  > kernel -1:
            x_imag[idx] = kernel -1

    ker = np.zeros((kernel, kernel))
    for idx in range(len(x_real)):
        ker[np.int(x_real[idx])][np.int(x_imag[idx])] = 1
    ker = ker/np.sum(np.sum(ker))
    return ker

def RandomMotionBlur(img, kernel):
    """
    add random mothon blur
    rft [Boracchi and Foi 2012] Giacomo Boracchi and Alessandro Foi, "Modeling the Performance of Image Restoration from Motion Blur"
    """
    kernel_motion_blur = CreateMotionKernel(kernel)
    dst = cv2.filter2D(img, -1, kernel_motion_blur)
    return dst

def AugImage(img):
    """
    Aug image randomly
    """
    Dices = np.random.randn(4,)
    BlurDice = np.random.randn()
    if Dices[0] > 0.8:
        img = Contrast(img)
    if Dices[1] > 0.8:
        img = Bright(img)
    if Dices[2] > 0.8:
        img = Saturation(img)
    if Dices[3] > 0.8:
        kerSize = np.int(np.random.randn()*5)
        kerSize = np.max([kerSize, 3])
        if BlurDice < 0.25:
           img = BoxBlur(img, 3)
        elif BlurDice < 0.5:
           kerSize = kerSize/2*2 + 1
           img = MedianBlur(img, kerSize)
        elif BlurDice < 0.75:
           kerSize = kerSize/2*2 + 1
           img = MotionBlur(img, kerSize)
    return img

def GrayWorld(img):
    """
    White balance algorithm
    """
    [h, w, c] = img.shape
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    K = 127.0
    R, G, B = cv2.split(RGB)
    Aver_R = np.average(np.average(R))
    Aver_G = np.average(np.average(G))
    Aver_B = np.average(np.average(B))
    #K =    (Aver_B +   Aver_G +   Aver_R)/3.0
    K_R =  K/Aver_R
    K_G =  K/Aver_G
    K_B =  K/Aver_B
    R = R * K_R
    G = G * K_G
    B = B * K_B
    R = np.where( R>255, 255,R)
    R = np.where( R<0, 0, R)
    G = np.where( G>255, 255,G)
    G = np.where( G<0, 0, G)
    B = np.where( B>255, 255,B)
    B = np.where( B<0, 0, B)
    RGB[:,:,0] = np.uint8(R)
    RGB[:,:,1] = np.uint8(G)
    RGB[:,:,2] = np.uint8(B)
    BGR = cv2.cvtColor(RGB, cv2.COLOR_RGB2BGR)
    return BGR


if __name__ == '__main__':
    image = cv2.imread('test11.jpg')
    im_vis = image.copy()
    image_blur = GrayWorld(image)
    cv2.imshow('blur', image_blur)
    cv2.waitKey(0)
    cv2.imshow('test', im_vis)
    cv2.waitKey(0)