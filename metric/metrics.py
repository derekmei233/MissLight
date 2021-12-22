# -*- coding:utf-8 -*-

import numpy as np
import torch

def masked_smape_np(y_true, y_pred, data_shape, mask_matrix, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.ones(data_shape)
        for idx, i in enumerate(mask_matrix):
            if i == 0:
                if len(data_shape) == 3:
                    mask[:, idx, :] = np.zeros_like(mask[:, idx, :])
                else:
                    mask[:, idx, :,:] = np.zeros_like(mask[:, idx, :,:])
        if len(data_shape) == 4:
            mask = mask.reshape(-1,1)
        mask = mask.astype('float32')
        # mask /= np.mean(mask)
        # mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
        #               y_true))
        smape = 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
        smape = np.nan_to_num(mask * smape)
        return np.mean(smape)

def smape_np(y_true, y_pred,null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        smape = 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
        smape = np.nan_to_num(smape)
        return np.mean(smape)


# preds,labels:(24,80,4,30)
def masked_mse(preds, labels, mask_matrix, null_val=np.nan):
    # if np.isnan(null_val):
    #     mask = ~torch.isnan(labels)
    # else:
    #     mask = (labels != null_val)
    mask = torch.ones_like(labels)
    for idx, i in enumerate(mask_matrix):
        if i == 0:
            mask[:, idx, :, :] = torch.zeros_like(mask[:, idx, :, :])
    mask = mask.float()
    # mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, mask_matrix, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, mask_matrix=mask_matrix,
                                 null_val=null_val))


def masked_mae(preds, labels, mask_matrix, null_val=np.nan):
    # if np.isnan(null_val):
    #     mask = ~torch.isnan(labels)
    # else:
    #     mask = (labels != null_val)
    mask = torch.ones_like(labels)
    for idx, i in enumerate(mask_matrix):
        if i == 0:
            mask[:, idx, :, :] = torch.zeros_like(mask[:, idx, :, :])
    mask = mask.float()
    # mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mae_test(y_true, y_pred, data_shape, mask_matrix, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        # if np.isnan(null_val):
        #     mask = ~np.isnan(y_true)
        # else:
        #     mask = np.not_equal(y_true, null_val)
        mask = np.ones(data_shape)
        for idx, i in enumerate(mask_matrix):
            if i == 0:
                if len(data_shape) == 3:
                    mask[:, idx, :] = np.zeros_like(mask[:, idx, :])
                else:
                    mask[:, idx, :,:] = np.zeros_like(mask[:, idx, :,:])
        if len(data_shape) == 4:
            mask = mask.reshape(-1,1)
        mask = mask.astype('float32')
        # mask /= np.mean(mask)
        mae = np.abs(np.subtract(y_pred, y_true).astype('float32'),
                     )
        mae = np.nan_to_num(mask * mae)
        return np.mean(mae)


def masked_rmse_test(y_true, y_pred, data_shape, mask_matrix, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        # if np.isnan(null_val):
        #     mask = ~np.isnan(y_true)
        # else:
        #     mask = np.not_equal(y_true, null_val)
        mask = np.ones(data_shape)
        for idx, i in enumerate(mask_matrix):
            if i == 0:
                if len(data_shape) == 3:
                    mask[:, idx, :] = np.zeros_like(mask[:, idx, :])
                else:
                    mask[:, idx, :,:] = np.zeros_like(mask[:, idx, :,:])
        if len(data_shape) == 4:
            mask = mask.reshape(-1,1)
        mask = mask.astype('float32')
        # mask /= np.mean(mask)
        mse = ((y_pred - y_true)**2)
        mse = np.nan_to_num(mask * mse)
        return np.sqrt(np.mean(mse))
