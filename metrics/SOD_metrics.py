import os
import time

import numpy as np
import torch
from torchvision import transforms
import scipy
import scipy.ndimage
from tqdm import tqdm
threshold_sal, upper_sal, lower_sal = 0.5, 1, 0


import numpy as np
from sklearn.metrics import confusion_matrix

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class SODMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, cuda=False):
        self.cuda = cuda
        self.mae_sum=0.0
        self.Fm_sum=0.0
        self.Sm_sum=0.0
        self.imgs_cnt=0
        self.times=0

    def update(self,  preds, labels):
        self.imgs_cnt +=labels.shape[0]
        self.times+=1
        self.cal_mae(preds, labels)
        # self.cal_Sm(preds, labels)
        # self.cal_Fm(preds, labels)

            
    def cal_mae(self, preds, labels):
        for pred, label in zip(preds, labels):
            mea = torch.abs(pred - label).mean()
            if mea == mea:  # for Nan
                self.mae_sum += mea
        
    def cal_Sm(self, preds, labels, alpha=0.5):
        avg_q, img_num = 0.0, 0  # alpha = 0.7; cited from the F-360iSOD
        for pred, gt in zip(preds, labels):
            gt[gt >= 0.5] = 1
            gt[gt < 0.5] = 0
            y = gt.mean()
            if y == 0:
                x = pred.mean()
                Q = 1.0 - x
            elif y == 1:
                x = pred.mean()
                Q = x
            else:
                # gt[gt>=0.5] = 1
                # gt[gt<0.5] = 0
                Q = alpha * self._S_object(pred, gt) + (1-alpha) * self._S_region(pred, gt)
                if Q.item() < 0:
                    Q = torch.FloatTensor([0.0])
            # img_num += 1
            # avg_q += Q.item()
            # if np.isnan(avg_q):
            #     raise #error
            if np.isnan(Q.item()):
                continue
            
            img_num += 1
            avg_q += Q.item()

        avg_q /= img_num
        self.Sm_sum+=avg_q
        
    
    def cal_Fm(self, preds, labels):
        beta2 = 0.3
        avg_f = 0.0  
        prec_avg=torch.zeros(255)
        recall_avg=torch.zeros(255)
        
        for pred, gt in zip(preds, labels):
            if self.cuda:
                prec_avg=prec_avg.cuda()
                recall_avg=recall_avg.cuda()
            
            # examples with totally black GTs are out of consideration
            if torch.mean(gt) == 0.0:
                continue

            prec, recall = self._eval_pr(pred, gt, 255)
            prec_avg+=prec
            recall_avg+=recall
            f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall+1e-20)
            f_score[f_score != f_score] = 0 # for Nan
            avg_f += f_score
            
        self.Fm_sum+=avg_f


    def _S_object(self, pred, gt):
        fg = torch.where(gt==0, torch.zeros_like(pred), pred)
        bg = torch.where(gt==1, torch.zeros_like(pred), 1-pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1-gt)
        u = gt.mean()
        Q = u * o_fg + (1-u) * o_bg

        return Q
    
    def _object(self, pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
        # if torch.isnan(score):
        #     raise

        return score
    
    def _S_region(self, pred, gt):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4
        # print(Q)

        return Q
    
    def _centroid(self, gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            if self.cuda:
                X = torch.eye(1).cuda() * round(cols / 2)
                Y = torch.eye(1).cuda() * round(rows / 2)
            else:
                X = torch.eye(1) * round(cols / 2)
                Y = torch.eye(1) * round(rows / 2)
        else:
            total = gt.sum()
            if self.cuda:
                i = torch.from_numpy(np.arange(0,cols)).cuda().float()
                j = torch.from_numpy(np.arange(0,rows)).cuda().float()
            else:
                i = torch.from_numpy(np.arange(0,cols)).float()
                j = torch.from_numpy(np.arange(0,rows)).float()
            X = torch.round((gt.sum(dim=0)*i).sum() / total)
            Y = torch.round((gt.sum(dim=1)*j).sum() / total)

        return X.long(), Y.long()
    
    def _eval_pr(self, y_pred, y, num):
        if self.cuda:
            prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

        return prec, recall
    
    def _divideGT(self, gt, X, Y):
        h, w = gt.size()[-2:]
        area = h*w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3

        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]

        return LT, RT, LB, RB

    def _ssim(self, pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h*w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x)*(pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y)*(gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x)*(gt - y)).sum() / (N - 1 + 1e-20)
        
        aplha = 4 * x * y *sigma_xy
        beta = (x*x + y*y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0

        return Q
   
    def get_results(self):
        """Returns accuracy score evaluation result.
            - mean_Fm
            - MAE
            - Sm
        """
        
        MAE=self.mae_sum/self.imgs_cnt
        # meanFm=(self.Fm_sum/self.imgs_cnt).mean().item()
        # Sm=self.Sm_sum/self.times
        # if self.cuda:
        #     # meanFm=meanFm.cpu().numpy()
        #     MAE=MAE.cpu().numpy()

        return {
                "MAE": MAE,
                # "mean_Fm": meanFm,
                # "Sm": Sm,
            }
        
    def reset(self):
        self.mae_sum=0.0
        self.Fm_sum=0.0
        self.Sm_sum=0.0
        self.imgs_cnt=0
