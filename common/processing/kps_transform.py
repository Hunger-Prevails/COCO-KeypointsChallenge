import numpy as np


def intersect_kps(ex_box, gt_kps):
    # ex_box :   (4,)   [x1, y1, x2, y2]
    # gt_kps :   (num_kps, 3)    [x, y, v]
    keep = np.where((ex_box[0] <= gt_kps[:, 0]) & (gt_kps[:, 0] <= ex_box[2]) &
                    (ex_box[1] <= gt_kps[:, 1]) & (gt_kps[:, 1] <= ex_box[3]))
    ex_kps = np.zeros(gt_kps.shape, dtype=gt_kps.dtype)
    ex_kps[keep, :] = gt_kps[keep, :]
    return ex_kps

def reloc_kps(src_box, src_kps, dst_box):
    # src_box :   (4,)   [x1, y1, x2, y2]
    # src_kps :   (num_kps, 3)   [x, y, v]
    # dst_box :   (4,)
    dst_kps = src_kps.copy()
    dst_kps[:, :2] = src_kps[:, :2] + src_box[:2] - dst_box[:2]
    return dst_kps

def resize_kps(src_box, src_kps, dst_box):
    # src_box :   (4,)   [x1, y1, x2, y2]
    # src_kps :   (num_kps, 3)   [x, y, v]
    # dst_box :   (4,)
    dst_kps = src_kps.copy()
    src_w = src_box[2] - src_box[0] + 1
    src_h = src_box[3] - src_box[1] + 1
    dst_w = dst_box[2] - dst_box[0] + 1
    dst_h = dst_box[3] - dst_box[1] + 1
    dst_kps[:, 0] = src_kps[:, 0] * ((dst_w - 1.) / (src_w - 1.)) if src_w > 1 else 0
    dst_kps[:, 1] = src_kps[:, 1] * ((dst_h - 1.) / (src_h - 1.)) if src_h > 1 else 0
    return dst_kps

def ravel_kps(src_kps, width):
    # src_kps :   (num_kps, 3)   [x, y, v]
    dst_kps = np.zeros((src_kps.shape[0],), dtype=src_kps.dtype)
    dst_kps[:] = src_kps[:, 1] * width + src_kps[:, 0]
    ignore = np.where(src_kps[:, 2] == 0)
    dst_kps[ignore] = -1
    return dst_kps

def unravel_kps(src_kps, width):
    # src_kps :   (num_kps, )
    dst_kps = np.ones((src_kps.shape[0], 3), dtype=src_kps.dtype)
    dst_kps[:, 0] = src_kps[:] % width
    dst_kps[:, 1] = src_kps[:] / width
    ignore = np.where(src_kps[:] == -1)
    dst_kps[ignore, :] = 0
    return dst_kps

def filter_kps(src_kps, src_box):
    keep = np.where((src_box[0] <= src_kps[:, 0]) & (src_kps[:, 0] <= src_box[2]) &
                    (src_box[1] <= src_kps[:, 1]) & (src_kps[:, 1] <= src_box[3]))
    dst_kps = np.zeros(src_kps.shape, dtype=src_kps.dtype)
    dst_kps[keep, :] = src_kps[keep, :]

def clip_kps(src_kps, src_box):
    x = src_kps[:, 0]
    y = src_kps[:, 1]
    x[x < src_box[0]] = src_box[0]
    x[x > src_box[2]] = src_box[2]
    y[y < src_box[1]] = src_box[1]
    y[y > src_box[3]] = src_box[3]
    dst_kps = np.zeros(src_kps.shape, dtype=src_kps.dtype)
    dst_kps[:, 0] = x
    dst_kps[:, 1] = y
    return dst_kps

debug_kps = False
def check_kps(kps, box):
    # kps :   (num_kps, 3)   [x, y, v]
    # box :   (4,)   [x1, y1, x2, y2]
    if debug_kps:
        for i in range(kps.shape[0]):
            x = kps[i][0]
            y = kps[i][1]
            v = kps[i][2]
            if v != 0:
                assert box[0] <= x <= box[2] and box[1] <= y <= box[3], \
                    'keypoint (%.2f, %.2f, %d) out of box (%.2f, %.2f, %.2f, %.2f)' % (x, y, v, box[0], box[1], box[2], box[3])

def kps2box(kps):
    # kps :   (num_kps, 3)   [x, y, v]
    box = np.zeros((4,), dtype=np.float32)
    kps = kps[:, :2]
    box[:2] = kps.min(axis=0)
    box[2:] = kps.max(axis=0)
    return box

def compute_kps_area(kps, im_height, im_width, box_area):
    # kps :   (num_kps, 3)   [x, y, v]
    from scipy.spatial import ConvexHull
    kps = clip_kps(kps, [0, 0, im_width-1, im_height-1])
    try:
        area = ConvexHull(kps[:, :2]).area
    except:
        area = box_area
    return area


_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
_oks_vars = (_oks_sigmas * 2) ** 2
def compute_oks(area_1, kps_1, kps_2):
    # kps_1 :   (num_kps, 3)   [x, y, v]
    # kps_2 :   (num_kps, 3)   [x, y, v]
    keep = np.where((kps_1[:, 2] >= 0.1) & (kps_2[:, 2] >= 0.1))[0]
    if len(keep) == 0:
        return 0
    else:
        dx = kps_1[:, 0] - kps_2[:, 0]
        dy = kps_1[:, 1] - kps_2[:, 1]
        e = (dx ** 2 + dy ** 2) / _oks_vars / (area_1 + np.spacing(1)) / 2
        e = e[keep]
        oks = np.sum(np.exp(-e)) / e.shape[0]
        return oks

def compute_dist(kps_1, kps_2):
    # kps_1 :   (num_kps, 3)   [x, y, v]
    # kps_2 :   (num_kps, 3)   [x, y, v]
    keep = np.where((kps_1[:, 2] >= 0.1) & (kps_2[:, 2] >= 0.1))[0]
    if len(keep) == 0:
        return 0
    else:
        dx = kps_1[:, 0] - kps_2[:, 0]
        dy = kps_1[:, 1] - kps_2[:, 1]
        e = (dx ** 2 + dy ** 2) / 2
        e = e[keep]
        oks = np.sum(np.exp(-e)) / e.shape[0]
        return oks


def kps_nms(all_boxes, all_kps, scores, im_height, im_width, keep_thresh, merge_thresh):
    # all_boxes :  (num_boxes, 4)
    # all_kps :  (num_boxes, num_kps, 3)
    # scores : (num_boxes, )
    areas = (all_boxes[:, 2] - all_boxes[:, 0] + 1) * (all_boxes[:, 3] - all_boxes[:, 1] + 1)
    order = scores.argsort()[::-1]
    keep = []
    merge = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.zeros((len(order)-1,))
        area = compute_kps_area(all_kps[i], im_height, im_width, areas[i])
        for j in range(ovr.shape[0]):
            ovr[j] = compute_oks(area, all_kps[i], all_kps[order[1+j]])
        merge_inds = np.where(ovr > merge_thresh)[0]
        merge.append(order[merge_inds + 1])
        keep_inds = np.where(ovr <= keep_thresh)[0]
        order = order[keep_inds + 1]
    return keep, merge

def merge_kps(all_kps):
    # all_kps:  (num_loss, num_kps, 3)
    dst_kps = np.zeros((all_kps.shape[1], all_kps.shape[2]), dtype=all_kps.dtype)  # (num_kps, 3)
    all_kps_scores = all_kps[:, :, 2]  # (num_loss, num_kps)
    kps_scores = np.sum(all_kps_scores, axis=0)  # (num_kps)
    offset_xy = all_kps[:, :, :2] * all_kps_scores[:, :, np.newaxis]  # (num_loss, num_kps, 2)
    dst_kps[:, :2] = np.sum(offset_xy, axis=0) / kps_scores[:, np.newaxis]
    dst_kps[:, 2] = kps_scores / all_kps.shape[0]
    return dst_kps

def kps_soft_nms(all_boxes, all_kps, scores, im_height, im_width, sigma=0.5, sfthresh=0.3, thresh=0.001, method=2):
    areas = (all_boxes[:, 2] - all_boxes[:, 0] + 1) * (all_boxes[:, 3] - all_boxes[:, 1] + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.zeros((len(order)-1,))
        new_scores = np.zeros((len(order)-1,))
        area = compute_kps_area(all_kps[i], im_height, im_width, areas[i])
        for j in range(ovr.shape[0]):
            ovr[j] = compute_oks(area, all_kps[i], all_kps[order[1+j]])  # similar to IOU
            ov = ovr[j]
            if method == 1:       # linear soft  NMS
                if ov > sfthresh:
                    weight = 1 - ov
                else:
                    weight = 1
            elif method == 2:     # gaussian soft NMS
                weight = np.exp(-ov**2/sigma)
            else:     # original NMS
                if ov > sfthresh:  # sfthresh similar to origin NMS threshold
                    weight = 0
                else:
                    weight = 1
            new_scores[j] = scores[order[1+j]] * weight

        inds = np.where(new_scores <= thresh)[0]
        order = order[inds + 1]
    return keep


def kps_transform(gt_kps, base_kps, scale=1.0):
    # gt_kps: (2,)
    # base_kps: (num_anchors, 2)
    kps_deltas = (gt_kps - base_kps) / scale
    return kps_deltas

def kps_pred(base_kps, kps_deltas, scale=1.0):
    # base_kps: (num_anchors, 2)
    # kps_deltas: (num_anchors, num_kps*2)
    dst_kps = np.zeros(kps_deltas.shape)
    kps_deltas *= scale
    dst_kps[:, 0::2] = base_kps[:, 0, np.newaxis] + kps_deltas[:, 0::2]
    dst_kps[:, 1::2] = base_kps[:, 1, np.newaxis] + kps_deltas[:, 1::2]
    return dst_kps









