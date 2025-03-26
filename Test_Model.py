import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from network.PCDEN import PCDEN



if __name__ == "__main__":
    print('Hello')

    input=torch.rand(1,3,512,512).cuda()
    print('input.shape:',input.shape)

    model=PCDEN().cuda()
    output=model(input)
    for x in output:
        print('x.shape:',x.shape)











