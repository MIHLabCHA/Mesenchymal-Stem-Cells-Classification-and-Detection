import os
import numpy as np
import torch
import nni
from torch import optim, nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score, roc_curve, auc
from utils.aux_funcs import reset_weights, oversampling_minority_indices, undersampling_majority_indices, calculate_fretchet, save_batch_img
from utils.model import InceptionV3


def T_scaling(logits, temperature):
    return torch.div(logits, temperature)


def kfold_train(with_NNI, logit_calibration, device, model, train_dataset, train_dataset_noaug, test_dataset, label_binarizer, criterion, folds=5, max_epochs=5, batch_size=64, learning_rate=1e-3, temper_lr=1e-3, weight_decay=1e-5, early_stop_threshold=5, early_stop_criterion='valid_loss', save_path=None, TTA=False, over_sampling=False, random_state=0):
    torch.manual_seed(random_state)
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)

    if logit_calibration:
        temperature = nn.Parameter(torch.ones(1).to(device))
        temper_criterion = nn.BCEWithLogitsLoss()
        temper_optimizer = optim.LBFGS([temperature], lr=temper_lr, max_iter=1e4, line_search_fn='strong_wolfe')

    outputs = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset, train_dataset.targets)):
        subsampler_gen = torch.Generator()
        if over_sampling:
            targets = np.array(train_dataset.targets)
            #train_ids = oversampling_minority_indices(train_ids, targets)
            #val_ids = oversampling_minority_indices(val_ids, targets)
            val_ids = undersampling_majority_indices(val_ids, targets)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids, subsampler_gen)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids, subsampler_gen)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=16)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=val_subsampler, num_workers=16)
        val_loader_noaug = torch.utils.data.DataLoader(train_dataset_noaug, batch_size=batch_size, sampler=val_subsampler, num_workers=16)
        
        model.apply(reset_weights)
        
        best_epoch = -1
        best_y_true_val, best_y_score_val = [], []
        best_temperature = 1
        if early_stop_criterion=='valid_loss':
            best_metric = float("inf")
        elif early_stop_criterion=='ap_score' or early_stop_criterion=='auc_score':
            best_metric = -1
        
        train_loss, val_loss = [], []
        for epoch in range(max_epochs):
            print("Fold: {} | Epoch: {} | Started ...".format(fold+1, epoch+1), end=' ')
            model.train()
            cur_loss, count = 0, 0
            for _, data in enumerate(train_loader, 0):
                img, label = data
                img, label = img.to(device), label.to(device)

                optimizer.zero_grad()
                if logit_calibration:
                    output = torch.div(model(img).squeeze(), temperature)
                else:
                    output = model(img)
                loss = criterion(output, label.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) # gradient clipping
                optimizer.step()
                
                cur_loss += loss.item()
                count += 1
                del loss
            train_loss_value = cur_loss/count
            train_loss.append(train_loss_value)
            
            # Validation
            model.eval()
            cur_loss, count = 0, 0
            if not TTA: # Normal validation
                y_true, y_logit, y_score = [], [], []
                with torch.no_grad():
                    for _, data in enumerate(val_loader_noaug, 0):
                        img, label = data
                        img, label = img.to(device), label.to(device)
                        if logit_calibration:
                            output = torch.div(model(img).squeeze(), temperature)
                        else:
                            output = model(img)
                        loss = criterion(output, label.long())
                        cur_loss += loss.item()
                        count += 1
                        ps = output.softmax(dim=1)
                        y_true.append(label.cpu().numpy())
                        y_logit.append(output.cpu().numpy())
                        y_score.append(ps.cpu().numpy())
                    y_true = np.concatenate(y_true, axis=0)
                    y_logit = np.concatenate(y_logit, axis=0)
                    y_score = np.concatenate(y_score, axis=0)
            else: # Validation with TTA (Test-Time Augmentation)
                y_true, y_logit_ori, y_score_ori = [], [], []
                with torch.no_grad():
                    # original images
                    subsampler_gen.manual_seed(random_state) # force subsampler to generate the same sequence
                    for _, data in enumerate(val_loader_noaug, 0):
                        img, label = data
                        img, label = img.to(device), label.to(device)
                        if logit_calibration:
                            output = torch.div(model(img).squeeze(), temperature)
                        else:
                            output = model(img)
                        loss = criterion(output, label.long())
                        cur_loss += loss.item()
                        count += 1
                        ps = output.softmax(dim=1)
                        y_true.append(label.cpu().numpy())
                        y_logit_ori.append(output.cpu().numpy())
                        y_score_ori.append(ps.cpu().numpy())
                    y_true = np.concatenate(y_true, axis=0)
                    y_logit_ori = np.concatenate(y_logit_ori, axis=0)
                    y_score_ori = np.concatenate(y_score_ori, axis=0)

                    # augmented images
                    y_logit_aug, y_score_aug = 0, 0
                    for _ in range(5):
                        y_logit_aug_tmp, y_score_aug_tmp = [], []
                        subsampler_gen.manual_seed(random_state) # force subsampler to generate the same sequence
                        for _, data in enumerate(val_loader, 0):
                            img, label = data
                            img, label = img.to(device), label.to(device)
                            if logit_calibration:
                                output = torch.div(model(img).squeeze(), temperature)
                            else:
                                output = model(img)
                            loss = criterion(output, label.long())
                            cur_loss += loss.item()
                            count += 1
                            ps = output.softmax(dim=1)
                            y_logit_aug_tmp.append(output.cpu().numpy())
                            y_score_aug_tmp.append(ps.cpu().numpy())
                        y_logit_aug_tmp = np.concatenate(y_logit_aug_tmp, axis=0)
                        y_score_aug_tmp = np.concatenate(y_score_aug_tmp, axis=0)
                        y_logit_aug += y_logit_aug_tmp
                        y_score_aug += y_score_aug_tmp
                    y_logit_aug += y_logit_ori
                    y_score_aug += y_score_ori
                    y_logit = y_logit_aug/6
                    y_score = y_score_aug/6
            
            val_loss_value = cur_loss/count
            val_loss.append(val_loss_value)
            y_true_onehot = label_binarizer.transform(y_true)
            epoch_auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='micro')
            epoch_ap = average_precision_score(y_true_onehot, y_score, average='micro')
            
            # Learn the temperature
            def _eval():
                temper_loss = temper_criterion(T_scaling(torch.tensor(y_logit.max(axis=1)).to(device), temperature), torch.tensor(y_true.astype(np.float32)).to(device))
                temper_loss.backward()
                return temper_loss
            if logit_calibration:
                temper_optimizer.step(_eval)

            if early_stop_criterion=='valid_loss':
                if val_loss_value<best_metric:
                    best_metric = val_loss_value
                    best_epoch = epoch
                    best_y_true_val = y_true
                    best_y_score_val = y_score
                    if logit_calibration:
                        best_temperature = temperature
                    else:
                        best_temperature = None
                    torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
                elif epoch-best_epoch>early_stop_threshold:
                    print("Early stopped at epoch {}".format(epoch+1))
                    break # start new fold
                print("Train loss: {:1.5f} | Valid loss: {:1.5f}".format(train_loss_value, val_loss_value))
            elif early_stop_criterion=='ap_score':
                if epoch_ap>best_metric:
                    best_metric = epoch_ap
                    best_epoch = epoch
                    best_y_true_val = y_true
                    best_y_score_val = y_score
                    if logit_calibration:
                        best_temperature = temperature
                    else:
                        best_temperature = None
                    torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
                elif epoch-best_epoch>early_stop_threshold:
                    print("Early stopped at epoch {}".format(epoch+1))
                    break # start new fold
                print("Train loss: {:1.5f} | Valid loss: {:1.5f} | AUPR: {:1.5f}".format(train_loss_value, val_loss_value, epoch_ap))
            elif early_stop_criterion=='auc_score':
                if epoch_auc>best_metric:
                    best_metric = epoch_auc
                    best_epoch = epoch
                    best_y_true_val = y_true
                    best_y_score_val = y_score
                    if logit_calibration:
                        best_temperature = temperature
                    else:
                        best_temperature = None
                    torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
                elif epoch-best_epoch>early_stop_threshold:
                    print("Early stopped at epoch {}".format(epoch+1))
                    break # start new fold
                print("Train loss: {:1.5f} | Valid loss: {:1.5f} | AUC: {:1.5f}".format(train_loss_value, val_loss_value, epoch_auc))
        
        if test_dataset is not None:
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
            model.load_state_dict(torch.load(os.path.join(save_path, "best_model.pth")))
            model.eval()
            cur_loss, count = 0, 0
            y_true_test, y_score_test = [], []
            with torch.no_grad():
                for _, data in enumerate(test_loader, 0):
                    img, label = data
                    img, label = img.to(device), label.to(device)
                    if logit_calibration:
                        output = torch.div(model(img).squeeze(), temperature)
                    else:
                        output = model(img)
                    loss = criterion(output, label.long())
                    cur_loss += loss.item()
                    count += 1
                    ps = output.softmax(dim=1)
                    y_true_test.append(label.cpu())
                    y_score_test.append(ps.cpu().numpy()) # in multiclass classification, get all proba (not max only)
                test_loss = cur_loss/count
                y_true_test = np.concatenate(y_true_test, axis=0)
                y_score_test = np.concatenate(y_score_test, axis=0)
        else:
            test_loss = None
            y_true_test = None
            y_score_test = None

        outputs.append((fold, train_loss, val_loss, test_loss, best_y_true_val, best_y_score_val, best_temperature, y_true_test, y_score_test, best_metric),)
        if with_NNI:
            nni.report_intermediate_result(best_metric)
        print('Fold {} ended ... best metric: {:1.5f}'.format(fold+1, best_metric))
    return outputs
