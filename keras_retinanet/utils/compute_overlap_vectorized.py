import numpy as np

# about 20x faster than compute_overlap as python function
def compute_overlap_vectorized(boxes, query_boxes):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """

    # get number of boxes
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    # pre-calculate areas of both boxes
    areaN = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    areaK = (query_boxes[:, 2] - query_boxes[:, 0]) * (query_boxes[:, 3] - query_boxes[:, 1])
    
    # allocate array of overlaps
    overlaps = np.zeros((N, K), dtype=np.float64)

    # process our comparison boxes, assumption that query_boxes is smaller than boxes
    for k in range(K):
        # check for x overlap
        iw = np.minimum(boxes[:, 2], query_boxes[k, 2]) - np.maximum(boxes[:, 0], query_boxes[k, 0])
        idxW = np.where(iw > 0)[0]
        
        # only process boxes we overlap wioth
        if len(idxW) > 0:
            # check for y overlap
            ih = np.minimum(boxes[idxW, 3], query_boxes[k, 3]) - np.maximum(boxes[idxW, 1], query_boxes[k, 1])
            idxH = np.where(ih > 0)[0]

            # only process if we have y overlap
            if len(idxH) > 0:
                # print(K, len(idxW), len(idxH))
                idxW = idxW[idxH]
                # subset x and y overlap info
                iw = iw[idxW]
                ih = ih[idxH]

                # get union area
                ua = np.float64(
                    areaN[idxW] +
                    areaK[k] - iw * ih
                )

                # calculate overlaps and update
                overlaps[idxW, k] = iw * ih / ua

    return overlaps