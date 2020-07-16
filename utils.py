"""
图像处理的一些常用函数

1. RGB模式下像素值范围是[0, 255]，因为RGB图片被保存为 8 bits unsigned integers
2. 寻找目标区域的两种方法：cropping or masking
3. 直方图用于显示像素强度（pixel value）的分布，直方图的 x 轴 为 bins (1D的情况下)
   对灰度图进行   直方图均衡化可改善图像的对比度
4. 图像模糊（平滑），blurring or smoothing, “mixture” of pixels in a neighborhood becomes our blurred pixel
   as the size of the kernel increases, the more blurred our image will become.
"""
import cv2, mahotas
import numpy as np
from matplotlib import pyplot as plt


def show_img(image, gray=False, mode='cv2'):
    '''
    :param camp: "gray" for gray scale
    '''
    if mode == 'cv2':
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    else:
        camp = 'gray' if gray else None
        plt.imshow(image, cmap=camp)
        plt.show()


def read_img(im_pth, gray=False, printable=False):
    '''
    :param im_pth: image path on the disk
    :return: a NumPy array representing the image.
    '''
    img = cv2.imread(filename=im_pth)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if printable:
        show_img(image=img, gray=gray)
        if gray:
            print("height  : %s pixels"
                  "\nwidth   : %s pixels"
                  "\nchannels: %s"%(img.shape[0], img.shape[1], 1))
        else:
            print("height  : %s pixels"
                  "\nwidth   : %s pixels"
                  "\nchannels: %s" % (img.shape[0], img.shape[1], img.shape[2]))
    return img


def write_img(image, img_pth):
    cv2.imwrite(img_pth, image)


def draw_shape(canvas, pt1, pt2, color=(0,0,255), thickness=1, mode='line'):
    '''
    :param canvas: an image canvas, where the shape will be drew on, (it could be a image)
    :param pt1: tuple with 2 elements (x, y), coordinates of begin point
    :param pt2:
                for line and rectangle: tuple with 2 elements (x, y), coordinates of end point
                for circle            : int number, set to 0 for drawing a point
    :param color: tuple with 3 elements which taking values in [0, 255] with RGB color space
    :param thickness: line thickness, set to -1 for filling the shape
    :param mode: draw mode --> ['line', 'rectangle']
    :return: drew img
    '''
    if mode == 'line':
        cv2.line(img=canvas, pt1=pt1, pt2=pt2, color=color, thickness=thickness)
    elif mode == 'rectangle':
        cv2.rectangle(img=canvas, pt1=pt1, pt2=pt2, color=color, thickness=thickness)
    elif mode == 'circle':
        cv2.circle(img=canvas, center=pt1, radius=pt2, color=color, thickness=thickness)
    return canvas


def translation(image, x, y):
    '''
    :param image:
    :param x, y: 用于构造translation matrix
                 x表示左右移动图像（沿着x轴），x为正值时向右移动x个像素点，负值时向左移动|x|个像素点
                 y表示上下移动图像（沿着y轴），y为正值时向下移动y个像素点，负值时向上移动|y|个像素点
    :return:
    '''
    show_img(image=image)
    M=np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0])) #(width, height)
    show_img(image=shifted)
    return shifted


def rotation(image, theta=None, center=None, scale=1.0):
    '''
    theta为正值时逆时针旋转图片theta度,theta为负值时顺时针旋转theta度
    :param image:
    :param theta: 旋转角度
    :param center: 旋转中心(width, heightt)
    :param scale: 缩放尺度，float
    :return:
    '''
    show_img(image=image)
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle=theta, scale=scale)# 中心, 旋转角度, 缩放尺度
    rotated = cv2.warpAffine(image, M, (w, h))
    show_img(image=rotated)
    return rotated

def resize(image, aspect_ratio=None, width=None, height=None, method=cv2.INTER_AREA):
    '''
    resize a image, 当width或height指定时计算缩放比例
    :param image:
    :param aspect_ratio: 缩放比例
    :param aspect_ratio: width after rotation
    :param aspect_ratio: height after rotation
    :param method: 差值方法
                   cv2.INTER_AREA
                   cv2.INTER_LINEAR
                   cv2.INTER_CUBIC
                   cv2.INTER_NEAREST
    :return:
    '''
    show_img(image=image)
    if aspect_ratio is None:
        if width is not None:
            aspect_ratio = width / image.shape[1]
            dim = (width, int(aspect_ratio * image.shape[0]))
        elif height is not None:
            aspect_ratio = height / image.shape[0]
            dim = (int(aspect_ratio * image.shape[1]) ,height)
    else:
        dim = (int(image.shape[1] * aspect_ratio), int(image.shape[0] * aspect_ratio)) #width, height
    resized = cv2.resize(image, dim, interpolation=method)
    show_img(image=resized)
    return resized

def flip(image, axis=0):
    '''
    flip an image around either the x or y axis, or even both.
    :param image:
    :param axis: 沿着哪个轴翻转，axis = [0, 1, -1]
                 0  --> 上下翻转（沿着x轴）
                 1  --> 水平翻转（沿着y轴）
                -1  --> 上下水平同时翻转
    :return:
    '''
    show_img(image=image)
    flipped = cv2.flip(image, axis)
    show_img(image=flipped)
    return flipped

def crop(image, start_x, end_x, start_y, end_y):
    show_img(image=image)
    cropped = image[start_y:end_y, start_x:end_x]#image shape is (heigt, width, depth)
    show_img(image=cropped)
    return cropped

def add(im1, im2):
    '''do clipping when the pixel is exceed [0,255]'''
    return cv2.add(im1,im2)

def subtract(im1,im2):
    return cv2.subtract(im1, im2)

def bitwise(im1, im2=None, mask=None, mode="NOT"):
    if mode == "AND":
        return cv2.bitwise_and(im1, im2, mask=mask)
    elif mode == "OR":
        return cv2.bitwise_or(im1, im2, mask=mask)
    elif mode == "XOR":
        return cv2.bitwise_xor(im1, im2, mask=mask)
    elif mode == "NOT":
        return cv2.bitwise_not(im1, mask=mask)

def _split_merge(image):
    (B,G,R) = cv2.split(image)
    print(image.shape)
    print(B.shape)
    print(G.shape)
    print(R.shape)
    zeros = np.zeros(image.shape[:2],dtype="uint8")
    cv2.imshow("Merge", cv2.merge([B, G, R]))
    cv2.imshow("Merge", cv2.merge([R, G, B]))# wrong
    cv2.imshow("Red", cv2.merge([zeros,zeros,R]))
    cv2.imshow("Green", cv2.merge([zeros,G,zeros]))
    cv2.imshow("Blue", cv2.merge([B,zeros,zeros]))
    cv2.waitKey(0)

def _color_space(image):
    cv2.imshow("Original", image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV", hsv)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    cv2.imshow("L*a*b*", lab)
    cv2.waitKey(0)

def gray_histogram(image, mask=None, bins=256):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist(images=[image],
                        channels=[0],
                        mask=mask, # 只考虑 mask 后得到目标区域的像素点
                        histSize=[bins], # number of bins
                        ranges=[0, 256])
    plt.figure()
    plt.title("Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

def color_histogram(image, mask=None, bins=256):
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("’Flattened’ Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist(images=[chan],
                            channels=[0],
                            mask=mask,
                            histSize=[bins],
                            ranges=[0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()

def multi_dim_histogram(image, dim=3, bins=32):
    if dim == 3:
        hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        print("\n3D_hist:\n%s"%hist)
    elif dim == 2:
        chans = cv2.split(image)
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1,3,1)
        hist = cv2.calcHist(images=[chans[1],chans[0]],
                            channels=[0, 1],
                            mask=None,
                            histSize=[bins, bins],
                            ranges=[0, 256, 0, 256])
        p = ax.imshow(hist, interpolation="nearest")
        ax.set_title("2D Color Histogram for G and B")
        plt.colorbar(p)

        ax = fig.add_subplot(1,3,2)
        hist = cv2.calcHist(images=[chans[1], chans[2]],
                            channels=[0, 1],
                            mask=None,
                            histSize=[bins, bins],
                            ranges=[0, 256, 0, 256])
        p = ax.imshow(hist, interpolation="nearest")
        ax.set_title("2D Color Histogram for G and R")
        plt.colorbar(p)

        ax = fig.add_subplot(1,3,3)
        hist = cv2.calcHist(images=[chans[0], chans[2]],
                            channels=[0, 1],
                            mask=None,
                            histSize=[bins, bins],
                            ranges=[0, 256, 0, 256])
        p = ax.imshow(hist, interpolation="nearest")
        ax.set_title("2D Color Histogram for B and R")
        plt.colorbar(p)

    plt.show()

def equalizeHist(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    cv2.imshow("Histogram Equalization", np.hstack([gray, eq]))
    cv2.waitKey(0)
    return gray

def blur(image, kernel_size=(3,3), mode='average', sigmax=0, sigmaC=21, sigmaS=21):
    '''
    :param image:
    :param kernel_size: odd number
    :param mode: 'average'  --> 中间像素点取其周围像素点的均值(不包括自己)
                 'gaussian' --> neighborhood pixels that are closer to the central pixel contribute more “weight” to the average.
                                there is a param called sigmaX, which is the standard deviation in the x-axis direction.
                                By setting this value to 0, we are instructing OpenCV to automatically compute them based on our kernel size
                 'median'   --> the median blur method has been most effective when removing salt-and-pepper noise
                                可以去除纹路细节和噪声
                 'bilateral' --> In order to reduce noise while still maintaining edges, we can use bilateral blurring,
                                 Bilateral blurring accomplishes this by introducing two Gaussian distributions.
                                 只考虑周围和自己像素值相近的像素点来进行模糊
    :return:
    '''
    if mode == 'average':
        blur = cv2.blur(image,ksize=kernel_size)
    elif mode == 'gaussian':
        blur = cv2.GaussianBlur(image,ksize=kernel_size,sigmaX=sigmax)
    elif mode == 'median':
        blur = cv2.medianBlur(image, ksize=kernel_size[0])
    elif mode == 'bilateral':
        blur = cv2.bilateralFilter(image,d=kernel_size[0],sigmaColor=sigmaC,sigmaSpace=sigmaS)
    cv2.imshow('blur', np.hstack((image,blur)))
    cv2.waitKey(0)
    return blur

def simple_threshold(image, threshold=125, max_value=255):
    '''
    convert a grayscale image to a binary image, where the pixels are either 0 or 255
    :param image:
    :return:
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T, binary = cv2.threshold(gray,thresh=threshold,maxval=max_value,type=cv2.THRESH_BINARY)
    T, binary_inv = cv2.threshold(gray,thresh=threshold,maxval=max_value,type=cv2.THRESH_BINARY_INV)
    show_img(np.hstack((gray, binary, binary_inv)))
    return binary

def adaptive_threshold(image,max_value=255,adaptive_mode=cv2.ADAPTIVE_THRESH_MEAN_C,threshold_mode=cv2.THRESH_BINARY_INV,neighbor=11,C_value=4):
    '''
    :param image:
    :param max_value:
    :param adaptive_mode: 阈值选取方法
                          cv2.ADAPTIVE_THRESH_MEAN_C
                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    :param threshold_mode: 二值化方法
                           cv2.THRESH_BINARY_INV
                           cv2.THRESH_BINARY
    :param neighbor: 近邻大小必须为奇数
    :param C_value: 用于调整阈值（阈值减去C_value）
    :return:
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray,
                                   maxValue=max_value,
                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   thresholdType=cv2.THRESH_BINARY_INV,
                                   blockSize=11, #neighborhood size, odd number
                                   C=4)
    cv2.imshow("Thresh", binary)
    cv2.waitKey(0)
    return binary

def otsu_threshold(image):
    '''
    Otsu’s method assumes there are two peaks in the grayscale histogram of the image. It then tries to find an optimal
    value to separate these two peaks – thus our threshold value of T.
    :param image:
    :return:
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = mahotas.thresholding.otsu(gray)
    print('The threshold is',T)
    binary = gray.copy()
    binary[binary > T] = 255
    binary[binary < 255] = 0
    binary = cv2.bitwise_not(binary)
    show_img(binary)
    return binary

def riddler_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = mahotas.thresholding.rc(gray)
    print('The threshold is',T)
    binary = gray.copy()
    binary[binary > T] = 255
    binary[binary < 255] = 0
    binary = cv2.bitwise_not(binary)
    show_img(binary)
    return binary



if __name__ == '__main__':
    img_pth = '../data/waves.jpg'
    image = read_img(img_pth)
    otsu_threshold(image)
    # blur(image,mode='bilateral',kernel_size=(7,7))
    # equalizeHist(image)
    # multi_dim_histogram(image,dim=2)
    # color_space(image)
    # split_merge(image)
    # resize(img_pth,width=500)
    # translation(img_pth)
    # img = read_img(img_pth, gray=False, printable=True)
    # img=draw_shape(img,pt1=(60, 60),pt2=0, color=(255,0,0),mode='circle',thickness=-1)
    # show_img(image=img, mode='plt')