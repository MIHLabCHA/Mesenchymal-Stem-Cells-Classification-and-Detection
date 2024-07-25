import cv2
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from scipy import linalg
from glob import glob
from torchvision.utils import make_grid, save_image


# Save variables
def save(filename, *args):
    # Get global dictionary
    glob = globals()
    d = {}
    for v in args:
        d[v] = glob[v]
    with open(filename, 'wb') as f:
        pickle.dump(d, f)


# Load variables
def load(filename):
    # Get global dictionary
    glob = globals()
    with open(filename, 'rb') as f:
        for k, v in pickle.load(f).items():
            glob[k] = v


# Reset model's weights
def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


# Freeze model's weights
def freeze_weights(model):
    cnt = 0
    for child in model.children():
        cnt += 1
        if cnt<9: # Freeze the first 8 layers
            for param in child.parameters():
                param.requires_grad = False


# Get pretrained dict ready for updating new model
def get_pretrained_dict_ready(pretrained_dict, new_model_dict):
    processed_dict = {}
    for k in new_model_dict.keys():
        decomposed_key = k.split('.')
        if ("fpn" not in decomposed_key) and ("regressionModel" not in decomposed_key) and ("classificationModel" not in decomposed_key):
            pretrained_key = "FE."+k
            processed_dict[k] = pretrained_dict[pretrained_key]
    return processed_dict


def get_pretrained_dict_ready_v2(pretrained_dict, new_model_dict):
    processed_dict = {}
    for k in new_model_dict.keys():
        decomposed_key = k.split('.')
        if ("fpn" not in decomposed_key) and ("regression_head" not in decomposed_key) and ("classification_head" not in decomposed_key):
            pretrained_key = "FE."+'.'.join(decomposed_key[2:])
            processed_dict[k] = pretrained_dict[pretrained_key]
    return processed_dict


# Save batched images
def save_batch_img(bimg, batch_size, save_path, epoch, mu, std):
    for b in range(bimg.shape[0]):
        bimg[b, :, :, :] = (bimg[b, :, :, :].detach().cpu().permute(1, 2, 0)*std+mu).permute(2, 0, 1)
    grid = make_grid(bimg, nrow=batch_size//2, normalize=True, value_range=(0, 1), scale_each=True)
    save_image(grid, os.path.join(save_path, 'bimg_{}.png'.format(epoch)))


# Oversampling minority indices
def oversampling_minority_indices(sample_ids, targets):
    """
    Oversample minority indices to reach the balanced status
    sample_ids: indices of samples in the dataset
    targets: targets of all samples in the dataset
    """
    sample_targets = np.array([targets[t] for t in sample_ids])
    class_counts = np.array([(sample_targets==t).sum() for t in np.unique(sample_targets)])
    new_sample_ids = []
    for idx in np.unique(sample_targets):
        class_ids = [t for t in sample_ids if targets[t]==idx]
        if class_counts[idx]<class_counts.max():
            class_ids_new = np.random.choice(class_ids, size=class_counts.max())
        else:
            class_ids_new = class_ids
        new_sample_ids.append(class_ids_new)
    new_sample_ids = np.concatenate(new_sample_ids, axis=0)
    new_sample_ids = np.sort(new_sample_ids)
    return new_sample_ids


# Under sampling majority indices
def undersampling_majority_indices(sample_ids, targets):
    """
    Undersample majority indices to reach the balanced status
    sample_ids: indices of samples in the dataset
    targets: targets of all samples in the dataset
    """
    sample_targets = np.array([targets[t] for t in sample_ids])
    class_counts = np.array([(sample_targets==t).sum() for t in np.unique(sample_targets)])
    new_sample_ids = []
    for idx in np.unique(sample_targets):
        class_ids = [t for t in sample_ids if targets[t]==idx]
        if class_counts[idx]>class_counts.min():
            class_ids_new = np.random.choice(class_ids, size=class_counts.min(), replace=False)
        else:
            class_ids_new = class_ids
        new_sample_ids.append(class_ids_new)
    new_sample_ids = np.concatenate(new_sample_ids, axis=0)
    new_sample_ids = np.sort(new_sample_ids)
    return new_sample_ids


# Interleave two 4D tensors along the channel dimension
def interleave_tensors(*args):
    """
    Inputs are 4D tensors of size [B, C, H, W] to be interleaved
    """
    stacked = torch.stack(list(args), dim=2)
    return torch.flatten(stacked, start_dim=1, end_dim=2)


# Make detections (detection model)
def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)):
        idx_list = []
        for idx, score in enumerate(preds[id]['scores']):
            if score>threshold: # select idx which meets the threshold
                idx_list.append(idx)
        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]
    return preds


# Color transfer
def image_stats(img):
    """
    img is an Lab image
    """
    (l, a, b) = cv2.split(img)
    (l_mean, l_std) = (l.mean(), l.std())
    (a_mean, a_std) = (a.mean(), a.std())
    (b_mean, b_std) = (b.mean(), b.std())
    return (l_mean, l_std, a_mean, a_std, b_mean, b_std)


def color_transfer(src, dst):
    """
    Transfer color from src to dst
    src: uint8 BGR image
    dst: uint8 GRAY image
    """
    src = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst = cv2.cvtColor(cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB).astype(np.float32)

    (l_mean_src, l_std_src, a_mean_src, a_std_src, b_mean_src, b_std_src) = image_stats(src)
    (l_mean_dst, l_std_dst, a_mean_dst, a_std_dst, b_mean_dst, b_std_dst) = image_stats(dst)

    (l, a, b) = cv2.split(dst)
    l -= l_mean_dst
    a -= a_mean_dst
    b -= b_mean_dst

    l = (l_std_dst/l_std_src)*l
    a = (a_std_dst/a_std_src)*a
    b = (b_std_dst/b_std_src)*b

    l += l_mean_src
    a += a_mean_src
    b += b_mean_src

    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return transfer


# Fretchet Inception Distance
def calculate_activation_stats(images, model, batch_size=128, dims=2048, cuda=False):
    model.eval()
    act = np.empty((len(images), dims))

    if cuda:
        batch = images.cuda()
    else:
        batch = images
    pred = model(batch)[0]

    # If model output is not scalar, apply global spatial average pooling
    if pred.size(2)!=1 or pred.size(3)!=1:
        pred = nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))
    
    act = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape==mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape==sigma2.shape, \
        'Training and test covariances have different dimensions'
    
    diff = mu1-mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0])*eps
        covmean = linalg.sqrtm((sigma1+offset).dot(sigma2+offset))
    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)

    return (diff.dot(diff)+np.trace(sigma1)+np.trace(sigma2)-2*tr_covmean)


def calculate_fretchet(images_real, images_fake, model):
    mu_1, std_1 = calculate_activation_stats(images_real, model, cuda=True)
    mu_2, std_2 = calculate_activation_stats(images_fake, model, cuda=True)

    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
    return fid_value


# Detection model evaluation
def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i] # predict
        # pred_boxes = output['boxes']
        # pred_scores = output['scores']
        # pred_labels = output['labels']

        true_positives = torch.zeros(output['boxes'].shape[0])   # 예측 객체 개수
 
        annotations = targets[sample_i]  # actual
        target_labels = annotations['labels'] if len(annotations) else []
        if len(annotations):    # len(annotations) = 3
            detected_boxes = []
            target_boxes = annotations['boxes']

            for pred_i, (pred_box, pred_label) in enumerate(zip(output['boxes'], output['labels'])): # 예측값에 대해서..

                # If targets are found break
                if len(detected_boxes) == len(target_labels): # annotations -> target_labels
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)   # box_index : 실제 어떤 바운딩 박스랑 IoU 가 가장 높은지 index
                if iou >= iou_threshold and box_index not in detected_boxes: # iou만 맞으면 통과?
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]  # 예측된거랑 실제랑 매핑해서 하나씩 index 채움
        batch_metrics.append([true_positives, output['scores'], output['labels']])
    return batch_metrics


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = torch.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = torch.unique(target_cls)   # 2가 거의 예측안됨

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = torch.cumsum(1 - tp[i],-1)
            tpc = torch.cumsum(tp[i],-1)

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = torch.tensor(np.array(p)), torch.tensor(np.array(r)), torch.tensor(np.array(ap))
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
