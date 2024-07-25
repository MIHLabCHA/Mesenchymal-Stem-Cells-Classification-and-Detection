# Load libraries
import os
import platform
import argparse
import pickle
import cv2
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from utils import model, custom_dataset, aux_funcs
from retinanet import model as detection_model
from retinanet.dataloader import collater, Resizer, Augmenter, Normalizer, UnNormalizer


def main(args=None):
    # Command-line arguments
    current_path = os.getcwd()

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', default='unspecified', help='input folder')
    parser.add_argument('-o', '--output', default=os.path.join(current_path, 'output'), help='output folder')
    parser.add_argument('-m', '--mode', default='sstack', help='sstack (singlestack) or mstack (multistack)')
    parser.add_argument('-a', '--arch', default='resnet50', help='classifier architecture')

    args = parser.parse_args()

    args.mode = args.mode.lower()
    args.arch = args.arch.lower()
    assert args.mode=='sstack' or args.mode=='mstack', 'expected: sstack or mstack, got: {}'.format(args.mode)
    assert args.arch=='alexnet' or args.arch=='inceptionv3' or args.arch=='resnet50' or args.arch=='shufflenetv2' or args.arch=='mobilenetv3', 'expected: alexnet, inceptionv3, resnet50, shufflenetv2, or mobilenetv3, got: {}'.format(args.arch)

    # Input folder
    if args.input=='unspecified':
        args.input = os.path.join(current_path, 'input', args.mode)

    print('- Get images from           : {}\n\
- Save outputs to           : {}\n\
- Input image type          : {}\n\
- Architecture of classifier: {}'.format(args.input, args.output, args.mode, args.arch))


    mu = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)


    # Create output folder
    if not os.path.exists(os.path.join(args.output, args.mode)):
        aux_funcs.create_folder(os.path.join(args.output, args.mode))


    # Create a data loader for test dataset
    test_loader = custom_dataset.create_test_loader(args.input, 300, mu, std)


    # Create model and load pretrained weights
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    classifier = model.CCCL(args.arch, False, 3, 4)
    classifier.load_state_dict(torch.load(os.path.join(current_path, 'pretrained_models', 'classification', '{}_{}.pth'.format(args.mode, args.arch)), map_location=device))
    classifier.to(device)
    classifier.eval()


    # Inference (classifier)
    print('Classification ...')
    sample_fname = []
    sample_pred = []
    base_idx = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader, 0):
            img, _ = data
            img = img.to(device)
            logit = classifier(img) # logits
            ps = logit.softmax(dim=1) # probabilities
            pred = torch.argmax(ps, dim=1) # predicted labels
            sample_pred.append(pred.cpu().numpy())
            # Get filenames
            for idx in range(img.shape[0]):
                fname, _ = test_loader.dataset.samples[base_idx+idx]
                if platform.system()=='Linux':
                    sample_fname.append(fname.replace(os.path.join(args.input, '0')+'/', ''))
                else: # Windows
                    sample_fname.append(fname.replace(os.path.join(args.input, '0')+'\\', ''))
            base_idx += img.shape[0]
    sample_pred = np.concatenate(sample_pred, axis=0).tolist()


    # Inference (detector)
    if args.mode=='sstack':
        print('Detection ...')
        bbox_list = []
        # Test dataset
        detection_test_loader = custom_dataset.create_detection_test_loader(os.path.join(args.input, '0'))
        unnormalizer = UnNormalizer()
        # Detection model
        retinanet = detection_model.resnet50(num_classes=1, pretrained=False)
        retinanet.load_state_dict(torch.load(os.path.join(current_path, 'pretrained_models', 'detection', 'sstack_retinanet.pth'), map_location=device))
        retinanet.to(device)
        retinanet.eval()
        with torch.no_grad():
            for idx, data in enumerate(detection_test_loader, 0):
                img = data['img'].to(device)
                scores, _, transformed_anchors = retinanet(img.float())
                #idxs = np.where(scores.cpu()>0.5)
                idxs = ()
                if scores.cpu().numpy().max()>0.5:
                    idxs += (np.argpartition(scores.cpu(), -1)[-1:],) # top-1 box

                bbox = []
                if len(idxs[0])!=0:
                    for j in range(idxs[0].shape[0]):
                        bbox.append(transformed_anchors[idxs[0][j], :].cpu().numpy())
                bbox = np.array(bbox)

                # Remove overlapping boxes
                if len(idxs[0])>1:
                    selected = []
                    for j in range(idxs[0].shape[0]):
                        selected.append(idxs[0][j])
                    idxs_clean = np.array(selected)
                    pairs = list(combinations(selected, 2))
                    for j in range(len(pairs)):
                        if aux_funcs.bbox_iou(bbox[pairs[j][0]], bbox[pairs[j][1]])>0.2:
                            if (idxs_clean==pairs[j][1].numpy()).any():
                                idxs_clean = np.delete(idxs_clean, np.where(idxs_clean==pairs[j][1].numpy())[0][0], 0)
                    idxs = (idxs_clean,)

                # Draw bounding box
                img = np.array(255*unnormalizer(data['img'][0, :, :, :])).copy()
                img[img<0] = 0
                img[img>255] = 255
                img = np.transpose(img, [1, 2, 0])
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_copy = img.copy()
                for j in range(idxs[0].shape[0]):
                    x1, y1, x2, y2 = int(bbox[idxs[0][j]][0]), int(bbox[idxs[0][j]][1]), int(bbox[idxs[0][j]][2]), int(bbox[idxs[0][j]][3])
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 255, 0), thickness=8)
                    bbox_list.append('{} {} {} {}'.format(x1, y1, x2, y2))
                if platform.system()=='Linux':
                    img_fname = detection_test_loader.dataset.files_grabbed[idx].replace(os.path.join(args.input, '0')+'/', '')
                else: # Windows
                    img_fname = detection_test_loader.dataset.files_grabbed[idx].replace(os.path.join(args.input, '0')+'\\', '')
                plt.imsave(os.path.join(args.output, args.mode, '{}_pred.jpg'.format(img_fname.split('.')[0])), img)


    # Save results to .csv file
    if args.mode=='sstack':
        csv_header = ['Input', 'Predicted label', 'Bounding box (x1 y1 x2 y2)']
        csv_data = tuple(zip(sample_fname, sample_pred, bbox_list))
        with open(os.path.join(args.output, args.mode, 'results.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            writer.writerows(csv_data)
            f.close()
    else:
        csv_header = ['Input', 'Predicted label']
        csv_data = tuple(zip(sample_fname, sample_pred))
        with open(os.path.join(args.output, args.mode, 'results.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            writer.writerows(csv_data)
            f.close()


    # Before exit
    custom_dataset.folder_reorganize(args.input)


if __name__=='__main__':
    main()