import cv2
import matplotlib as matplotlib
import torch
from skimage import io,color,exposure,morphology,transform,img_as_float
import matplotlib.pyplot as plt
import skimage.transform as st
import numpy as np
import os
from PIL import Image, ImageOps
import pytesseract
import shutil
import cv2 as cv
import time
import copy

def rotate_bound(image, angle):
    # 获取宽高
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # 提取旋转矩阵 sin cos
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像的新边界尺寸
    nW = int((h * sin) + (w * cos))
    #     nH = int((h * cos) + (w * sin))
    nH = h

    # 调整旋转矩阵
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def get_minAreaRect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    io.imsave(r'gray.jpg', thresh)
    coords = np.column_stack(np.where(thresh > 0))
    return cv2.minAreaRect(coords)

def tagRMB(imgPath, tpl):
    target = cv.imread(imgPath)

    th, tw = tpl.shape[:2]
    #     res=cv.matchTemplate(target,tpl,cv.TM_CCOEFF_NORMED)
    res = cv.matchTemplate(target, tpl, cv.TM_CCOEFF)

    # axis=1,行方向
    res_sorted = sorted(res.max(axis=1), reverse=True)
    res_dif = [0] * 150
    for i in range(150):
        res_dif[i] = (res_sorted[i] - res_sorted[i + 1]) * 100. / res_sorted[i + 1]

    max_lastIdx = res_dif.index(sorted(res_dif, reverse=True)[0])

    idx = np.argwhere(res >= res_sorted[max_lastIdx])
    idx_set = set(np.unique(idx[:, 0]))

    for i in range(len(idx)):
        if idx[i, 0] in idx_set:
            idx_set.remove(idx[i, 0])
            tl = (idx[i, 1], idx[i, 0])
            br = (tl[0] + tw, tl[1] + th)
            cv.rectangle(target, tl, br, (0, 0, 0), 1)

    cv.imwrite(imgPath, target)

# 膨胀腐蚀操作
def dil2ero(img, selem):
    img = morphology.dilation(img, selem)
    imgres = morphology.erosion(img, selem)
    return imgres

# 收缩点团为单像素点（3×3）
def isolate(img):
    idx = np.argwhere(img < 1)
    rows, cols = img.shape

    for i in range(idx.shape[0]):
        c_row = idx[i, 0]
        c_col = idx[i, 1]
        if c_col + 1 < cols and c_row + 1 < rows:
            img[c_row, c_col + 1] = 1
            img[c_row + 1, c_col] = 1
            img[c_row + 1, c_col + 1] = 1
        if c_col + 2 < cols and c_row + 2 < rows:
            img[c_row + 1, c_col + 2] = 1
            img[c_row + 2, c_col] = 1
            img[c_row, c_col + 2] = 1
            img[c_row + 2, c_col + 1] = 1
            img[c_row + 2, c_col + 2] = 1
    return img

# 将图像边框变白
def clearEdge(img, width):
    img[0:width - 1, :] = 1
    img[1 - width + 1:-1, :] = 1
    img[:, 0:width - 1] = 1
    img[:, 1 - width:-1] = 1
    return img

# 创建文件夹
def mkdir(path):
    import os

    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print('文件夹创建成功')
        return True
    else:
        print('文件夹已存在')
        shutil.rmtree(path)
        os.makedirs(path)
        print('已清空文件夹')
        return False

# 分割图像
def splitImg(imgFilePath, split_pic_path):
    '''
    imgFilePath，图片路径
    split_pic_path，目标文件夹路径
    '''
    shutil.rmtree(tempDir)
    mkdir(tempDir)

    # 读取图片，并转灰度图
    img = io.imread(imgFilePath, True)
    # 二值化
    img_forSplit = copy.deepcopy(img)
    img_temp = copy.deepcopy(img)
    # img 提取边框用
    bi_th = img.max() * 0.875
    img[img < bi_th] = 0
    img[img >= bi_th] = 1
    # img_forSplit 分割用
    bi_th = 0.8
    img_forSplit[img_forSplit < bi_th] = 0
    img_forSplit[img_forSplit >= bi_th] = 1

    io.imsave(deskpath + 'img_forSplit.jpg', img_forSplit)
    #######################################################
    # 求图像中的横线和竖线
    rows, cols = img.shape
    # scale = 50

    col_selem = morphology.rectangle(cols // 30, 1)
    io.imsave(deskpath + '666.jpg', col_selem)
    img_cols = dil2ero(img, col_selem)
    io.imsave(deskpath+'img_cols.jpg',img_cols)

    row_selem = morphology.rectangle(1, rows // 50)
    img_rows = dil2ero(img, row_selem)
    io.imsave(deskpath+'img_rows.jpg',img_rows)

    ########################################################
    img_line = img_cols * img_rows
    io.imsave(deskpath+'img_line.jpg',img_line)

    # col和row的图里的线有点短。。。求dot图的时候刚好没重叠。。
    # 先延长，再求dot图
    idx_img_rows = np.argwhere(img_rows == 0)
    img_rows_temp = img_rows
    for i in range(idx_img_rows.shape[0]):
        img_rows_temp[
            idx_img_rows[i, 0],
            idx_img_rows[i, 1] + 1 if idx_img_rows[i, 1] + 1 < cols else idx_img_rows[i, 1]] = 0

    img_dot = img_cols + img_rows_temp
    io.imsave(deskpath + 'img_dot0.jpg', img_dot)
    img_dot[img_dot > 0] = 1
    io.imsave(deskpath + 'img_dot1.jpg', img_dot)
    img_dot = clearEdge(img_dot, 3)
    io.imsave(deskpath + 'img_dot2.jpg', img_dot)
    img_dot = isolate(img_dot)
    io.imsave(deskpath+'img_dot.jpg',img_dot)

    ########################################################
    # 获取表格顶点位置idx
    idx = np.argwhere(img_dot < 1)
    for n in range(idx.shape[0] - 1):
        if (abs(idx[n, 0] - idx[n + 1, 0]) < 5):
            idx[n + 1, 0] = idx[n, 0]

    for p in range(idx.shape[0] - 1):
        for q in range(idx.shape[0] - 1):
            if (idx[p, 1] < idx[q, 1]):
                tempp = idx[p, 0]
                tempq = idx[p, 1]
                idx[p, 0] = idx[q, 0]
                idx[p, 1] = idx[q, 1]
                idx[q, 0] = tempp
                idx[q, 1] = tempq

    for n in range(idx.shape[0] - 1):
        if (abs(idx[n, 1] - idx[n + 1, 1]) < 3):
            idx[n + 1, 1] = idx[n, 1]
    print(idx)
    idx_unirow = np.unique(idx[:, 0])
    # idx_unirow = idx[:, 0]
    print(idx_unirow)
    # 一行一行的来处理各个点

    # 保存计数器
    countHere = 0
    for i in range(idx_unirow.shape[0] - 1):
        # 当前行号、下一行行号、中间行号
        r_cur = idx_unirow[i]
        r_next = idx_unirow[i + 1]
        r_mid = (r_cur + r_next) // 2

        idx_currow = idx[idx[:, 0] == r_cur]
        idx_nextrow = idx[idx[:, 0] == r_next]
        print(idx_currow)

        # 遍历当前行的前n-1个点
        for j in range(idx_currow.shape[0] - 1):
            # 当左上角顶点下没有line的时候，则不是一个单元格的起始顶点
            if (idx_currow[j + 1, 1] == idx_currow[j, 1]):
                continue

            gflag = 0
            for t in range(-4, 4, 1):
                if (img_line[r_mid, idx_currow[j, 1] + t] != 1):
                    gflag = 1
                    break
            if (gflag == 0):
                continue

            offset = 1
            bottom_c = 0
            while (j + offset < idx_currow.shape[0]):
                # 找单元格的右上角顶点
                flag = 0
                for t in range(-4, 4, 1):
                    if (img_line[r_mid, idx_currow[j + offset, 1] + t] != 1):
                        flag = 1
                        break
                if (flag == 0):
                    offset = offset + 1
                else:
                    bottom_c = idx_currow[j + offset, 1]
                    break

            if bottom_c == 0:
                continue

            print(bottom_c)
            print(idx_nextrow[:, 1])
            idx_temp = idx_nextrow[idx_nextrow[:, 1] == bottom_c]
            if (idx_temp.shape[0] > 0):
                imghere = img_temp[r_cur:r_next, idx_currow[j, 1]:bottom_c]
                countHere += 1
                path = split_pic_path + '\\' + '{0:0>4}_{1:0>6}'.format(i, countHere) + '.png'
                print(path)
                io.imsave(path, imghere)


def focusImg(imgPath):
    img = io.imread(imgPath)
    img = color.rgb2gray(img)
    img = img_as_float(img)
    img = clearEdge(img, 5)

    # 求各列的和
    col_sum = img.sum(axis=0)
    # 求各行的和
    row_sum = img.sum(axis=1)

    idx_col_sum = np.argwhere(col_sum < col_sum.max())
    if len(idx_col_sum) == 0:
        os.remove(imgPath)
        return
    col_start, col_end = idx_col_sum[0, 0] - 1, idx_col_sum[-1, 0] + 2

    idx_row_sum = np.argwhere(row_sum < row_sum.max())
    if len(idx_row_sum) == 0:
        os.remove(imgPath)
        return
    row_start, row_end = idx_row_sum[0, 0] - 1, idx_row_sum[-1, 0] + 2

    # 覆盖源文件保存
    io.imsave(imgPath, img[row_start:row_end, col_start:col_end])
#     sizeStr=str(row_end-row_start)+"×"+str(col_end-col_start)
#     return sizeStr

def getHProjection(image):
    hProjection = np.zeros(image.shape, np.uint8)
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像高度一致的数组
    h_ = [0] * h
    # 循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
    # 绘制水平投影图像
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y, x] = 255
    cv2.imshow('hProjection2', hProjection)

    return h_


def getVProjection(image):
    vProjection = np.zeros(image.shape, np.uint8);
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像宽度一致的数组
    w_ = [0] * w
    # 循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y, x] == 255:
                w_[x] += 1
    # 绘制垂直平投影图像
    for x in range(w):
        for y in range(h - w_[x], h):
            vProjection[y, x] = 255
    cv2.imshow('vProjection',vProjection)
    return w_

def splitChar(imge):
    img_temp = copy.deepcopy(imge)
    origineImage = imge
    # 图像灰度化
    # image = cv2.imread('test.jpg',0)
    image = cv2.cvtColor(origineImage, cv2.COLOR_BGR2GRAY)
    # 将图片二值化
    retval, img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    # 图像高与宽
    (h, w) = img.shape
    Position = []
    # 水平投影
    H = getHProjection(img)

    start = 0
    H_Start = []
    H_End = []
    # 根据水平投影获取垂直分割位置

    for i in range(len(H)):
        if H[i] > 0 and start == 0:
            H_Start.append(i)
            start = 1
        if H[i] <= 0 and start == 1:
            H_End.append(i)
            start = 0
    # 分割行，分割之后再进行列分割并保存分割位置
    print(len(H_Start))
    for i in range(len(H_Start)):
        # 获取行图像
        print(i)
        cropImg = img[H_Start[i]:H_End[i], 0:w]
        # cv2.imshow('cropImg',cropImg)
        # cv2.waitKey(0)
        # 对行图像进行垂直投影
        W = getVProjection(cropImg)
        Wstart = 0
        Wend = 0
        W_Start = 0
        W_End = 0
        for j in range(len(W)):
            if W[j] > 0 and Wstart == 0:
                W_Start = j
                Wstart = 1
                Wend = 0
            if W[j] <= 0 and Wstart == 1:
                W_End = j
                Wstart = 0
                Wend = 1
            if Wend == 1:
                Position.append([W_Start, H_Start[i], W_End, H_End[i]])
                Wend = 0
    images = []
    for m in range(len(Position)):
        cropImg = img_temp[Position[m][1]:Position[m][3],Position[m][0]:Position[m][2]]
        images.append(cropImg)
        cv2.imshow('abc', cropImg)
        cv2.rectangle(origineImage, (Position[m][0], Position[m][1]), (Position[m][2], Position[m][3]), (0, 229, 238), 1)
    cv2.imshow('image', origineImage)
    cv2.waitKey(0)
    return images


def my_main(srcFilePath,txtpath,fnameIn,tempdir):

    print('开始分割单元格')
    # 分割单元格，二值化
    splitImg(srcFilePath,tempdir)

    print('开始focus')
    # focus，覆盖原图
    for fpath,fdir,fs in os.walk(tempdir):
        for f in fs:
            filepath=os.path.join(fpath,f)
            focusImg(filepath)
#     Tesseract-OCR
    pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'
    with open(txtpath, 'w') as ftxt:
        for fpath, fdir, fs in os.walk(tempdir):
            for f in fs:
                filepath = os.path.join(fpath, f)
                fname, fext = os.path.splitext(f)
                img = Image.open(filepath)
                print(fname + ":")
                ftxt.write(fname + ":")
                content = pytesseract.image_to_string(img, lang='chi_sim')
                print(content)
                ftxt.write(content)
            ftxt.write('\n')

if __name__ == '__main__':
    saveDir = r'saveDir'
    tplpath=r'pic.jpg'
    tpl=cv.imread(tplpath)
    tfileDir=r'tfile'
    tempDir=r'temp'
    res_txtDir=r'res_txt'
    countf=0
    txtpath=os.path.join(res_txtDir,'res.txt')
    fname=r'res'
    deskpath=r'a'
    my_main(tplpath,txtpath,fname,tempDir)
    print('\n跑完了')

