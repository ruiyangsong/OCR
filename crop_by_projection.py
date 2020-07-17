from utils import *
global max_value
max_value = 255

def main():
    # img_pth = '../data/ZSK42357-000117-L000059.tif'
    img_pth = '../data/ZSK42357-000009-L000005.tif'
    # img_pth = '../data/ZSK42357-000004-L000002.ti f'
    # img_pth = '../data/ZSK42357-000003-L000002.tif'
    # img_pth = '../data/ZSK42357-000002-L000001.tif'
    binary = load_processing(img_pth)
    binary_cols = crop_col(binary)

    # for binary_col in binary_cols:
    #     characters = crop_row(binary_col) #文字array组成的list


def crop_col(binary):
    #先垂直投影按列切割
    min_thresh = 20 #波峰的最小幅度
    min_range = 80 #两个波峰的最小间隔
    w_w = vProject(binary)
    begin = 0
    end   = 0
    w_begin, w_end = [], []
    # start = 0
    for j in range(len(w_w)):
        if w_w[j] > min_thresh and begin == 0:
            begin = j
        elif w_w[j] > min_thresh and begin != 0:
            continue
        elif w_w[j] < min_thresh and begin != 0:
            end = j
            if end - begin >= min_range:
                w_begin.append(begin)
                w_end.append(end)
                begin = 0
                end = 0
        elif w_w[j] < min_thresh or begin != 0:
            continue
    print('w_begin',w_begin)
    print('w_end',w_end)

    binary_cols = []
    for begin, end in zip(w_begin, w_end):
        binary_col = binary[:,begin:end]
        binary_cols.append(binary_col)
        cv2.imshow('cols',binary_col)
        cv2.waitKey(0)

        crop_row(binary_col)
    return binary_cols


def crop_row(binary_col):
    #水平投影按列切割
    min_thresh = 20 #波峰的最小幅度
    min_range = 60 #两个波峰的最小间隔
    h_h = hProject(binary_col)
    begin = 0
    end   = 0
    h_begin, h_end = [], []
    # start = 0
    for j in range(len(h_h)):
        if h_h[j] > min_thresh and begin == 0:
            begin = j
        elif h_h[j] > min_thresh and begin != 0:
            continue
        elif h_h[j] < min_thresh and begin != 0:
            end = j
            if end - begin >= min_range:
                h_begin.append(begin)
                h_end.append(end)
                begin = 0
                end = 0
        elif h_h[j] < min_thresh or begin != 0:
            continue
    print('h_begin',h_begin)
    print('h_end',h_end)

    characters = []
    for begin, end in zip(h_begin, h_end):
        characters.append(binary_col[begin:end])
        cv2.imshow('character',binary_col[begin:end])
        cv2.waitKey(0)
    return characters

        # if w_w[j] > 0 and start == 0:
        #     w_start.append(j)
        #     start = 1
        # if w_w[j] == 0 and start == 1:
        #     w_end.append(j)
        #     start = 0

    #水平投影按行切割

    # h_h = hProject(binary)
    #
    # start = 0
    # h_start, h_end = [], []
    # position = []
    #
    # # 根据水平投影获取垂直分割
    # for i in range(len(h_h)):
    #     if h_h[i] > 0 and start == 0:
    #         h_start.append(i)
    #         start = 1
    #     if h_h[i] == 0 and start == 1:
    #         h_end.append(i)
    #         start = 0
    #
    # for i in range(len(h_start)):
    #     cropImg = th[h_start[i]:h_end[i], 0:w]
    #     if i == 0:
    #         cv2.imshow('cropimg', cropImg)
    #         cv2.imwrite('words_cropimg.jpg', cropImg)
    #     w_w = vProject(cropImg)
    #
    #     wstart, wend, w_start, w_end = 0, 0, 0, 0
    #     for j in range(len(w_w)):
    #         if w_w[j] > 0 and wstart == 0:
    #             w_start = j
    #             wstart = 1
    #             wend = 0
    #         if w_w[j] == 0 and wstart == 1:
    #             w_end = j
    #             wstart = 0
    #             wend = 1
    #
    #         # 当确认了起点和终点之后保存坐标
    #         if wend == 1:
    #             position.append([w_start, h_start[i], w_end, h_end[i]])
    #             wend = 0
    #
    # # 确定分割位置
    # for p in position:
    #     cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 2)
    #
    # cv2.imshow('image', img)
    # cv2.imshow('th', th)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def load_processing(img_pth):
    image = read_img(img_pth)
    show_img(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = otsu_threshold(gray, max_value=max_value,printable=True)
    binary = cv2.bitwise_not(binary)
    binary = erode(binary, ksize=3, printable=True)  # 膨胀
    # black character on white paper
    print('The binary shape is:', binary.shape)
    return binary

# 水平方向投影
def hProject(binary):
    h, w = binary.shape

    # 水平投影的亮度分布图
    hprojection = np.zeros(binary.shape, dtype=np.uint8)

    # 创建h长度都为0的数组
    h_h = [0]*h
    for j in range(h):#对第 j 行, 统计黑色的个数
        for i in range(w):
            if binary[j,i] == 0:
                h_h[j] += 1
    # 画出投影图
    for j in range(h):
        for i in range(h_h[j]):
            hprojection[j,i] = 255

    cv2.imshow('hpro', hprojection)
    cv2.waitKey(0)

    return h_h

# 垂直反向投影
def vProject(binary):
    h, w = binary.shape
    # 垂直投影
    vprojection = np.zeros(binary.shape, dtype=np.uint8)

    # 创建 w 长度都为0的数组
    w_w = [0]*w
    for i in range(w):
        for j in range(h):
            if binary[j, i ] == 0:
                w_w[i] += 1

    for i in range(w):
        for j in range(w_w[i]):
            vprojection[j,i] = 255

    cv2.imshow('vpro', vprojection)
    cv2.waitKey(0)
    return w_w


if __name__ == '__main__':
    main()
