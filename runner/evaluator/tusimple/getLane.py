import cv2
import numpy as np

def isShort(lane):
    start = [i for i, x in enumerate(lane) if x > 0]
    if not start:
        return 1
    else:
        return 0

def fixGap(coordinate):
    if any(x > 0 for x in coordinate):
        start = [i for i, x in enumerate(coordinate) if x > 0][0]
        end = [i for i, x in reversed(list(enumerate(coordinate))) if x > 0][0]
        lane = coordinate[start:end+1]
        if any(x < 0 for x in lane):
            gap_start = [i for i, x in enumerate(
                lane[:-1]) if x > 0 and lane[i+1] < 0]
            gap_end = [i+1 for i,
                       x in enumerate(lane[:-1]) if x < 0 and lane[i+1] > 0]
            gap_id = [i for i, x in enumerate(lane) if x < 0]
            if len(gap_start) == 0 or len(gap_end) == 0:
                return coordinate
            for id in gap_id:
                for i in range(len(gap_start)):
                    if i >= len(gap_end):
                        return coordinate
                    if id > gap_start[i] and id < gap_end[i]:
                        gap_width = float(gap_end[i] - gap_start[i])
                        lane[id] = int((id - gap_start[i]) / gap_width * lane[gap_end[i]] + (
                            gap_end[i] - id) / gap_width * lane[gap_start[i]])
            if not all(x > 0 for x in lane):
                print("Gaps still exist!")
            coordinate[start:end+1] = lane
    return coordinate

def getLane_tusimple(prob_map, y_px_gap, pts, thresh, resize_shape=None, cfg=None):
    """
    Arguments:
    ----------
    prob_map: prob map for single lane, np array size (h, w)
    resize_shape:  reshape size target, (H, W)

    Return:
    ----------
    coords: x coords bottom up every y_px_gap px, 0 for non-exist, in resized shape
    """
    if resize_shape is None:
        resize_shape = prob_map.shape
    h, w = prob_map.shape
    H, W = resize_shape
    H -= cfg.cut_height

    coords = np.zeros(pts)
    coords[:] = -1.0
    for i in range(pts):
        y = int((H - 10 - i * y_px_gap) * h / H)
        if y < 0:
            break
        line = prob_map[y, :]
        id = np.argmax(line)
        if line[id] > thresh:
            coords[i] = int(id / w * W)
    if (coords > 0).sum() < 2:
        coords = np.zeros(pts)
    fixGap(coords)
    return coords


def prob2lines_tusimple(seg_pred, exist, resize_shape=None, smooth=True, y_px_gap=10, pts=None, thresh=0.3, cfg=None):
    """
    Arguments:
    ----------
    seg_pred:      np.array size (5, h, w)
    resize_shape:  reshape size target, (H, W)
    exist:       list of existence, e.g. [0, 1, 1, 0]
    smooth:      whether to smooth the probability or not
    y_px_gap:    y pixel gap for sampling
    pts:     how many points for one lane
    thresh:  probability threshold

    Return:
    ----------
    coordinates: [x, y] list of lanes, e.g.: [ [[9, 569], [50, 549]] ,[[630, 569], [647, 549]] ]
    """
    if resize_shape is None:
        resize_shape = seg_pred.shape[1:]  # seg_pred (5, h, w)
    _, h, w = seg_pred.shape
    H, W = resize_shape
    coordinates = []

    if pts is None:
        pts = round(H / 2 / y_px_gap)

    seg_pred = np.ascontiguousarray(np.transpose(seg_pred, (1, 2, 0)))
    for i in range(cfg.num_classes - 1):
        prob_map = seg_pred[..., i + 1]
        if smooth:
            prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
        coords = getLane_tusimple(prob_map, y_px_gap, pts, thresh, resize_shape, cfg)
        if isShort(coords):
            continue
        coordinates.append(
            [[coords[j], H - 10 - j * y_px_gap] if coords[j] > 0 else [-1, H - 10 - j * y_px_gap] for j in
             range(pts)])


    if len(coordinates) == 0:
        coords = np.zeros(pts)
        coordinates.append(
            [[coords[j], H - 10 - j * y_px_gap] if coords[j] > 0 else [-1, H - 10 - j * y_px_gap] for j in
             range(pts)])


    return coordinates
