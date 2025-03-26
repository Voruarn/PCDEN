from tqdm import tqdm
import utils
import os
import random
import argparse
import numpy as np
import sys

from torch.utils import data

from network.PCDEN import PCDEN
import torch.nn.functional as F
import torch
import torch.nn as nn
from datasets.ActiveFireTestset import ActiveFireDataset

from PIL import Image
import collections


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')

    p8 = im.convert("P")  # 将24位深的RGB图像转化为8位深的模式“P”图像
    p8.save(d_dir+image_name+'.png')


def get_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--trainset_path", type=str, 
        default='',
        help="path to Dataset")
    parser.add_argument("--testset_path", type=str, 
        default='',
        help="path to Dataset")
    
    parser.add_argument("--dataset", type=str, default='ActiveFire', help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help='num_classes')
    parser.add_argument("--backbone", type=str, default='resnet50',
        help='model name:[resnet50, VGG, mobilenet_v2]')
    parser.add_argument("--mid_ch", type=int, default=128,
                        help="mid_ch")
    parser.add_argument("--model", type=str, default='PCDEN',
        help='model name:[]')
    parser.add_argument("--batch_size", type=int, default=1,
                        help='batch size ')
    parser.add_argument("--trainsize", type=int, default=256)

    parser.add_argument("--n_cpu", type=int, default=1,
                        help="download datasets")
    
    parser.add_argument("--ckpt", type=str,
            default='',
              help="restore from checkpoint")
   
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--pred_path", type=str, default='',
                        help="random seed (default: 1)")

    return parser


def get_dataset(opts):
    val_dst = ActiveFireDataset(is_train=False,voc_dir=opts.testset_path)
    return val_dst

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    

    opts = get_argparser().parse_args()
    
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    
    print("Device: %s" % device)

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    test_dst = get_dataset(opts)
    
    print('opts:',opts)
  
    test_loader = data.DataLoader(
        test_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu)
    print("Dataset: %s, Test set: %d" %
          (opts.dataset, len(test_dst)))

    model = eval(opts.model)(mid_ch=opts.mid_ch)
    
    opts.model+='_w'+str(opts.mid_ch)
    opts.pred_path+=opts.model+'/'
    if not os.path.exists(opts.pred_path):
        os.makedirs(opts.pred_path, exist_ok=True)
    
    print('opts:',opts)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        try:
            model.load_state_dict(checkpoint["model_state"])
            print('try: load pth from:', opts.ckpt)
        except:
            dic = collections.OrderedDict()
            for k, v in checkpoint["model_state"].items():
                #print( k)
                mlen=len('module')+1
                newk=k[mlen:]
                # print(newk)
                dic[newk]=v
            model.load_state_dict(dic)
            print('except: load pth from:', opts.ckpt)

        model=model.to(device)  
    else:
        print("[!] Retrain")
        model.to(device)


    model.eval()
    data_loader = tqdm(test_loader, file=sys.stdout)

    
    for batch in data_loader:
        imgs,  name=batch['img'], batch['name']

        imgs = imgs.to(device, dtype=torch.float32)
       
        S= model(imgs)
        pred=torch.sigmoid(S[0])
        pred = pred.squeeze()

        pred = normPRED(pred)
    
        save_output(name[0], pred, opts.pred_path)
          

if __name__ == '__main__':
    main()
