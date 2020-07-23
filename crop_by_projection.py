#!/usr/bin/env python
'''
Crop the picture to individual characters by horizontal and vertical projections
'''
from utils import *
import os, argparse

global max_value
global verbose
max_value = 255
verbose   = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_pth',          type=str,   required=True, help='path of the input image')
    parser.add_argument('--out_dir',         type=str,   default='.',   help='directory of the ouput image')
    parser.add_argument('--first',           type=str,   choices=('V','H'), default='V', help='crop column or row first, dafault is "V"')
    parser.add_argument('--min_thresh_rate', type=float, default=0.2,   help='minimum peak thresh_rate to median in the histogram, default is 0.2')
    parser.add_argument('--iter_open',       type=int,   default=3,     help='iterations of open operation, default is 3')
    parser.add_argument('--iter_dilate',     type=int,   default=20,    help='iterations of dilate operation, default is 20')

    args = parser.parse_args()
    img_pth         = args.im_pth
    out_dir         = args.out_dir
    first           = args.first
    min_thresh_rate = args.min_thresh_rate
    iter_open       = args.iter_open
    iter_dilate     = args.iter_dilate

    #
    # begins here
    #
    os.makedirs(out_dir, exist_ok=True)
    print('Command line parameters'
          '\nimg_pth        : %s'
          '\nout_dir        : %s'
          '\nfirst          : %s'
          '\nmin_thresh_rate: %s'
          '\niter_open      : %s'
          '\niter_dilate    : %s'%(img_pth, out_dir, first, min_thresh_rate, iter_open, iter_dilate))

    binary = load_processing(img_pth)
    write_img(binary, img_pth='%s/binary.png' % out_dir)

    if first == 'V':
        print('\n@corp column')
        binary_cols, vprojection, binary_copy = crop_col(binary, iter_open=iter_open, iter_dilate=iter_dilate, min_thresh_rate=min_thresh_rate)
        write_img(vprojection, img_pth='%s/vprojection.png'%out_dir)
        write_img(binary_copy, img_pth='%s/processing.png' %out_dir)
        for col_idx in range(len(binary_cols)):
            print('\ncolumn %s'%col_idx)
            col_out_dir = os.path.join(out_dir, 'col'+str(col_idx))
            os.makedirs(col_out_dir, exist_ok=True)
            binary_col = binary_cols[col_idx]
            write_img(binary_col, img_pth='%s/col%s.png' % (col_out_dir, col_idx))
            # crop row
            characters, hprojection, binary_col_copy = crop_row(binary_col, iter_open=iter_open, iter_dilate=iter_dilate, min_thresh_rate=min_thresh_rate)
            write_img(hprojection, img_pth='%s/col%s_hprojection.png' % (col_out_dir,col_idx))
            write_img(binary_col_copy, img_pth='%s/col%s_processing.png' % (col_out_dir, col_idx))
            for row_idx in range(len(characters)):
                write_img(characters[row_idx], img_pth='%s/col%s_row%s.png'%(col_out_dir, col_idx, row_idx))

    elif first == 'H':
        binary_rows, hprojection, binary_col_copy = crop_row(binary, iter_open=iter_open, iter_dilate=iter_dilate, min_thresh_rate=min_thresh_rate)
        write_img(hprojection, img_pth='%s/hprojection.png' % out_dir)
        write_img(binary_col_copy, img_pth='%s/processing.png' % out_dir)
        for row_idx in range(len(binary_rows)):
            print('\nrow %s' % row_idx)
            row_out_dir = os.path.join(out_dir, 'row'+str(row_idx))
            os.makedirs(row_out_dir, exist_ok=True)
            binary_row = binary_rows[row_idx]
            write_img(binary_row, img_pth='%s/row%s.png' % (row_out_dir, row_idx))
            # crop col
            charactres, vprojection, binary_copy = crop_col(binary_row, iter_open=iter_open, iter_dilate=iter_dilate, min_thresh_rate=min_thresh_rate)
            write_img(vprojection, img_pth='%s/row%s_vprojection.png' % (row_out_dir, row_idx))
            write_img(binary_copy, img_pth='%s/row%s_processing.png' % (row_out_dir, row_idx))
            for col_idx in range(len(charactres)):
                write_img(charactres[col_idx], img_pth='%s/row%s_%s.png'%(row_out_dir, row_idx, col_idx))


def load_processing(img_pth):
    image = read_img(img_pth)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # black characters
    binary = otsu_threshold(gray, max_value=max_value) # black characters
    print('\nThe binary shape is:', binary.shape)
    return binary

def crop_col(binary, iter_open, iter_dilate, min_thresh=20, min_range=0, min_thresh_rate=None):
    '''
    :param binary:
    :param min_thresh: 波峰的最小幅度
    :param min_range: 两个波峰的最小间隔
    :param min_thresh_rate: 当此参数不为0时，自适应计算 min_thresh
    :return: 垂直切割后的子图组成的list [从左到右排序]
    '''
    #
    #预处理
    #
    binary_copy = binary.copy()
    binary_copy = cv2.bitwise_not(opening(cv2.bitwise_not(binary_copy), kernel=np.ones((1, 3), dtype='uint8'), iter=iter_open))  # 去除竖线
    binary_copy = cv2.bitwise_not(dilate(cv2.bitwise_not(binary_copy), kernel=np.ones((3, 1), dtype='uint8'), iter=iter_dilate)) # 纵向膨胀

    #
    # 投影
    #
    w_w, vprojection = vProject(binary_copy)
    if min_thresh_rate is not None:
        min_thresh = min_thresh_rate * max(w_w)

    #
    #计算间隔的中位数
    #
    begin = 0
    w_begin, w_end = [], []
    for j in range(len(w_w)):
        if w_w[j] > min_thresh and begin == 0:
            begin = j
        elif w_w[j] < min_thresh and begin != 0:
            end = j
            if end - begin >= min_range and w_w[j - 1] < min_thresh:
                begin = j
            elif end - begin >= min_range and w_w[j - 1] >= min_thresh:
                w_begin.append(begin)
                w_end.append(end)
                begin = 0

    #
    # 以中位数为 min_range 重新计算 w_begin, w_end
    #
    col_width_lst = [w_end[i] - w_begin[i] for i in range(len(w_end))]
    min_range = np.median(col_width_lst)*0.8
    begin = 0
    w_begin, w_end = [], []
    for j in range(len(w_w)):
        if w_w[j] > min_thresh and begin == 0:
            begin = j
        elif w_w[j] < min_thresh and begin != 0:
            end = j
            if end - begin >= min_range and w_w[j-1] < min_thresh:
                begin = j
            elif end - begin >= min_range and w_w[j-1] >= min_thresh:
                w_begin.append(begin)
                w_end.append(end)
                begin = 0

    #
    #对太宽的的进行切割
    #
    w_begin_new, w_end_new = [], []
    col_width_lst = [w_end[i] - w_begin[i] for i in range(len(w_end))]
    median_width = np.median(col_width_lst)
    for i in range(len(w_end)):
        ratio = int(round(col_width_lst[i] / median_width))
        if ratio == 0:
            pass
        elif ratio == 1:
            w_begin_new.append(w_begin[i])
            w_end_new.append(w_end[i])
        elif ratio > 1:
            sub_width = round(col_width_lst[i]/ratio)
            w_begin_new.append(w_begin[i])
            w_end_new.append(w_begin[i]+sub_width)
            for _ in range(ratio-1):
                w_begin_new.append(w_end_new[-1])
                w_end_new.append(w_begin_new[-1]+sub_width)

    print('w_w        :', w_w)
    print('w_begin    :', w_begin)
    print('w_end      :', w_end)
    print('width      :', list(map(lambda x, y: x - y, w_end, w_begin)))
    print('w_begin_new:', w_begin_new)
    print('w_end_new  :', w_end_new)
    print('width_new  :', list(map(lambda x, y: x - y, w_end_new, w_begin_new)))

    binary_cols = []
    for begin, end in zip(w_begin_new, w_end_new):
        binary_col = binary[:,begin:end]
        binary_cols.append(binary_col)

    return binary_cols, vprojection, binary_copy


def crop_row(binary_col, iter_open, iter_dilate, min_thresh=20, min_range=0, min_thresh_rate=None):
    #
    # 预处理
    #
    binary_col_copy = binary_col.copy()
    binary_col_copy = cv2.bitwise_not(erode(cv2.bitwise_not(binary_col_copy), kernel=np.ones((3, 1), dtype='uint8'), iter=iter_open))  # 去除字间的上下连接
    binary_col_copy = cv2.bitwise_not(dilate(cv2.bitwise_not(binary_col_copy), kernel=np.ones((1, 3), dtype='uint8'), iter=iter_dilate))  # 横向膨胀

    #
    # 水平投影
    #
    h_h, hprojection = hProject(binary_col_copy)
    if min_thresh_rate is not None:
        min_thresh = min_thresh_rate * max(h_h)

    #
    # 计算高度的中位数
    #
    begin = 0
    h_begin, h_end = [], []
    for j in range(len(h_h)):
        if h_h[j] > min_thresh and begin == 0:
            begin = j
        elif h_h[j] < min_thresh and begin != 0:
            end = j
            if end - begin >= min_range and h_h[j - 1] < min_thresh:
                begin = j
            elif end - begin >= min_range and h_h[j - 1] >= min_thresh:
                h_begin.append(begin)
                h_end.append(end)
                begin = 0

    #
    # 以中位数为 min_range 重新计算 h_begin, h_end
    #
    row_height_lst = [h_end[i] - h_begin[i] for i in range(len(h_end))]
    min_range = np.median(row_height_lst) * 0.8
    begin = 0
    h_begin, h_end = [], []
    for j in range(len(h_h)):
        if h_h[j] > min_thresh and begin == 0:
            begin = j
        elif h_h[j] < min_thresh and begin != 0:
            end = j
            if end - begin >= min_range and h_h[j - 1] < min_thresh:
                begin = j
            elif end - begin >= min_range and h_h[j - 1] >= min_thresh:
                h_begin.append(begin)
                h_end.append(end)
                begin = 0

    #
    # 对太高的行进行切割
    #
    h_begin_new, h_end_new = [], []
    row_height_lst = [h_end[i] - h_begin[i] for i in range(len(h_end))]
    median_height = np.median(row_height_lst)
    for i in range(len(h_end)):
        ratio = int(round(row_height_lst[i] / median_height))
        if ratio == 0:
            pass
        elif ratio == 1:
            h_begin_new.append(h_begin[i])
            h_end_new.append(h_end[i])
        elif ratio > 1:
            sub_height = round(row_height_lst[i] / ratio)
            h_begin_new.append(h_begin[i])
            h_end_new.append(h_begin[i] + sub_height)
            for j in range(ratio - 1):
                h_begin_new.append(h_end_new[-1])
                h_end_new.append(h_begin_new[-1] + sub_height)

    print('h_h        :', h_h)
    print('h_begin    :', h_begin)
    print('h_end      :', h_end)
    print('height     :', list(map(lambda x, y: x - y, h_end, h_begin)))
    print('h_begin_new:', h_begin_new)
    print('h_end_new  :', h_end_new)
    print('height_new :', list(map(lambda x, y: x - y, h_end_new, h_begin_new)))

    characters = []
    for begin, end in zip(h_begin_new, h_end_new):
        character = binary_col[begin:end, :]
        characters.append(character)
    return characters, hprojection, binary_col_copy

def hProject(binary):
    h, w = binary.shape
    h_h = [0]*h # 每行黑色像素点的个数
    for j in range(h):
        for i in range(w):
            if binary[j,i] == 0:
                h_h[j] += 1

    hprojection = np.zeros(binary.shape, dtype=np.uint8)
    for j in range(h):
        for i in range(h_h[j]):
            hprojection[j,i] = 255
    return h_h, hprojection

def vProject(binary):
    h, w = binary.shape
    w_w = [0]*w # 每列黑色像素点的个数
    for i in range(w):
        for j in range(h):
            if binary[j, i ] == 0:
                w_w[i] += 1

    vprojection = np.zeros(binary.shape, dtype=np.uint8)
    for i in range(w):
        for j in range(w_w[i]):
            vprojection[j,i] = 255
    return w_w, vprojection

if __name__ == '__main__':
    main()
