from utils import *
global max_value
max_value = 255

def main():
    # img_pth = '../data/ZSK42357-000117-L000059.tif'
    # img_pth = '../data/ZSK42357-000009-L000005.tif'
    # img_pth = '../data/ZSK42357-000004-L000002.tif'
    # img_pth = '../data/ZSK42357-000003-L000002.tif'
    img_pth = '../data/010600b.tif'
    # img_pth = '../data/ZSK42357-000002-L000001.tif'
    binary = load_processing(img_pth)

    # edges = sobel(binary)
    # edges = cv2.bitwise_not(edges)
    # row_index, col_index = calc_edges(edges)

    row_index, col_index = calc_edges(binary)

    for j in range(len(col_index)-1):
        row_up = row_index[0]
        row_down = row_index[1]
        col_left = col_index[j]
        col_right = col_index[j+1]

        col_binary = binary[row_up:row_down,col_left:col_right]
        # cv2.imshow("Croped", col_binary)
        # cv2.waitKey(0)

        crop(col_binary)


def load_processing(img_pth):
    image = read_img(img_pth)
    show_img(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = otsu_threshold(gray, max_value=max_value,printable=True)
    binary = cv2.bitwise_not(binary)
    binary = erode(binary, ksize=3, printable=True)  # 膨胀
    # write_img(binary, '../data/010600b_____.tif')
    # black character on white paper
    print('The binary shape is:', binary.shape)
    return binary

def calc_edges(binary):
    h,w = binary.shape
    min_gap_row = h/20
    min_gap_col = 100#w/20
    sum_row = np.sum(binary,axis=0)#对行求和
    latent_col = [index for index in range(w) if sum_row[index] < h * max_value * 0.5]
    col_index = [latent_col[0]]
    gaps = []
    for col in latent_col[1:]:
        if col - col_index[-1] > min_gap_col:
            gaps.append(col - col_index[-1])
            col_index.append(col)

    median = np.median(gaps)
    print('latent col_index', latent_col)
    print('median',median)
    print('old col_index', col_index)

    for j in range(1, len(col_index)):
        if col_index[j] - col_index[j-1] > median*1.5:
            col_index.insert(j, int(col_index[j-1] + (col_index[j] - col_index[j-1])/2))

    print('col_index', col_index)

    sum_col = np.sum(binary,axis=1)#对列求和
    latent_row = [index for index in range(h) if sum_col[index] < w * max_value * 0.5]
    print('latent row index',latent_row)
    row_index = [latent_row[0]]
    for row in latent_row[1:]:
        if row - row_index[-1] > min_gap_row:
            row_index.append(row)
    print('row_index',row_index)

    return row_index, col_index



def crop(col):
    cv2.imshow("Croped", col)
    sum_row = np.sum(col, axis=0)
    borders = np.argwhere(sum_row > max_value*0.98*col.shape[0]).reshape(-1)
    border_dist = []
    for j in range(1,len(borders)):
        border_dist.append(borders[j]-borders[j-1])
    border_l = borders[np.argmax(border_dist)]
    border_r = borders[np.argmax(border_dist)+1]
    print(col.shape, borders, border_l, border_r)

    col_binary = col[10:-10, border_l:border_r]
    cv2.imshow("Croped", col_binary)
    cv2.waitKey(0)

    h,w = col_binary.shape
    min_gap_row = h/20
    sum_col = np.sum(col_binary,axis=1)
    latent_row = [index for index in range(h) if sum_col[index] > w * max_value * 0.98]
    row_index = [latent_row[0]]
    for row in latent_row[1:]:
        if row - row_index[-1] > min_gap_row:
            row_index.append(row)
    if row_index[0] !=0:
        row_index.insert(0,0)
    print('row_index', row_index)

    for i in range(len(row_index)-1):
        row_up = row_index[i]
        row_down = row_index[i+1]
        character = col_binary[row_up:row_down]
        if np.sum(character)/(character.shape[0]*character .shape[1]*max_value) < 0.9:
            cv2.imshow("Croped%s" % i, character)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()