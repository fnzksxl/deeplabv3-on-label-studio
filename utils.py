import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def return_mask_label_name(mask,label_map):
    object_masks = []
    to_Seperate_mask = mask.astype(np.uint8)

    for label in np.unique(mask):
        if label == 0:
            continue  # Skip background (assuming background is labeled as 0)
        
        # Create a binary mask for the current label
        label_mask = (to_Seperate_mask == label).astype(np.uint8)

        # Find contours of the binary mask
        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        for contour  in contours:
            label_num = to_Seperate_mask[contour[0][0][1], contour[0][0][0]]
            print("Label Number >>> ")
            print(label_num)
            if label_num in np.array(list(label_map.keys()))+1:
                object_mask = np.zeros_like(mask,dtype=np.uint8)
                cv2.drawContours(object_mask,[contour],0,1,-1)
                
                object_masks.append((object_mask, label_num))
    
    return object_masks

def mask_to_polygons(mask):
    polygons = []
    for label in np.unique(mask):
        if label == 0:
            continue  # Skip background (assuming background is labeled as 0)
        
        # Create a binary mask for the current label
        label_mask = (mask == label).astype(np.uint8)

        # Find contours of the binary mask
        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = 0.0005 * cv2.arcLength(contour, True)
            approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
            polygons.append((approx_polygon,label))

    return polygons

def normalize_polygons(polygons, image_size):
    width, height = image_size
    normalized_polygons = []
    polygons=polygons.tolist()
    for polygonss in polygons:
      for point in polygonss:
          normalized_x = (point[0] / width) * 100
          normalized_y = (point[1] / height) * 100
          normalized_polygons.append([normalized_x, normalized_y])

    return normalized_polygons

def create_mask_from_polygons(polygons,label,image_size):
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    for polygon in polygons:
        polygon_np = np.array(polygon, np.int32)
        polygon_np = polygon_np.reshape((-1, 1, 2))

        cv2.fillPoly(mask, [polygon_np], label)

    return mask

def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), ignore_index=ignore_index, size_average=False)
    loss = criterion(logit, target.long())

    if size_average:
        loss /= (h * w)

    if batch_average:
        loss /= n

    return loss

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)

def decode_seg_map_sequence(label_masks):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def decode_segmap(label_mask, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    n_classes = 2
    label_colours = np.asarray([[0,0,0],[128,0,0]])
    
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def get_iou(pred, gt, n_classes=2):
    total_iou = 0.0
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]

        intersect = [0] * n_classes
        union = [0] * n_classes
        for j in range(n_classes):
            match = (pred_tmp == j) + (gt_tmp == j)

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un

        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        img_iou = (sum(iou) / len(iou))
        total_iou += img_iou

    return total_iou