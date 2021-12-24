# -*- coding:utf-8 -*-

import numpy as np
import torch

def masked_mape_test(y_true, y_pred, data_shape, mask_matrix, null_val=np.nan):
    # with np.errstate(divide='ignore', invalid='ignore'):
    y_true_ = y_true + np.float32(y_true == 0)
    mask = np.ones(data_shape)
    for idx, i in enumerate(mask_matrix):
        if i == 0:
            if len(data_shape) == 3:
                mask[:, idx, :] = np.zeros_like(mask[:, idx, :])
            else:
                mask[:, idx, :,:] = np.zeros_like(mask[:, idx, :,:])
    mask = mask.astype('float32')
    mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                  y_true_))
    # mape = np.fabs((y_true[valid]-y_pred[valid])/y_true[valid])
    mape = mask * mape
    return np.mean(mape)

'''
preds,labels:(24,80,11,30)
loss except road related to virtual inter
mask_matrix:
    0:virtual node
    1:unmasked node
    2:masked node
'''

def masked_mse(preds, labels, mask_matrix, type, null_val=np.nan):
    mask = torch.ones_like(labels)
    for idx, i in enumerate(mask_matrix):
        if type == 'train':
            if i == 0 or i == 2:
                mask[:, idx, :, :] = torch.zeros_like(mask[:, idx, :, :])
        else:
            if i == 0:
                mask[:, idx, :, :] = torch.zeros_like(mask[:, idx, :, :])
    mask = mask.float()
    # mask /= torch.mean((mask))
    # mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    # loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mae_test(y_true, y_pred, data_shape, mask_matrix, null_val=np.nan):
    # with np.errstate(divide='ignore', invalid='ignore'):
    mask = np.ones(data_shape)
    for idx, i in enumerate(mask_matrix):
        if i == 0:
            if len(data_shape) == 3:
                mask[:, idx, :] = np.zeros_like(mask[:, idx, :])
            else:
                mask[:, idx, :,:] = np.zeros_like(mask[:, idx, :,:])
    # if len(data_shape) == 4:
    #     mask = mask.reshape(-1,1)
    mask = mask.astype('float32')
    # mask /= np.mean(mask)
    mae = np.abs(np.subtract(y_pred, y_true).astype('float32'))
    mae = mask * mae
    # mae = np.nan_to_num(mask * mae)
    return np.mean(mae)


def masked_rmse_test(y_true, y_pred, data_shape, mask_matrix, null_val=np.nan):
    # with np.errstate(divide='ignore', invalid='ignore'):
    mask = np.ones(data_shape)
    for idx, i in enumerate(mask_matrix):
        if i == 0:
            if len(data_shape) == 3:
                mask[:, idx, :] = np.zeros_like(mask[:, idx, :])
            else:
                mask[:, idx, :,:] = np.zeros_like(mask[:, idx, :,:])
    # if len(data_shape) == 4:
    #     mask = mask.reshape(-1,1)
    mask = mask.astype('float32')
    # mask /= np.mean(mask)
    mse = ((y_pred - y_true)**2)
    mse = mask * mse
    # mse = np.nan_to_num(mask * mse)
    return np.sqrt(np.mean(mse))
