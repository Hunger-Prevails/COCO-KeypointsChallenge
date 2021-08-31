import init_paths
import mxnet as mx
import cv2
import numpy as np


def read_list(path_imglist):
    imglist = {}
    with open(path_imglist) as fin:
        for line in iter(fin.readline, ''):
            line = line.strip().split('\t')
            imglist[line[-1]] = int(line[0])
    return imglist

def main():
    img_name = 'val2017'
    root_dir = '/opt/hdfs/user/xinze.chen/common/dataset/coco2017/images_lst_rec/'
    img_dir = '/data/xinze.chen/common/dataset/coco2017/images/%s/' % img_name
    path_imgidx = root_dir + '%s.idx' % img_name
    path_imgrec = root_dir + '%s.rec' % img_name
    path_imglist = root_dir + '%s.lst' % img_name

    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    imglist = read_list(path_imglist)

    i = 0
    for img_name in imglist:
        i += 1
        if i % 100 == 0:
            print '%d/%d' % (i, len(imglist))
        _, img = mx.recordio.unpack_img(imgrec.read_idx(imglist[img_name]), cv2.IMREAD_COLOR)
        img1 = cv2.imread(img_dir + img_name, cv2.IMREAD_COLOR)
        assert np.sum(abs(img - img1)) == 0














if __name__ == '__main__':
    main()
