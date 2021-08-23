import time
import cv2
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import torch
import torch.nn as nn

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output

def attempt_load(weights):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        #attempt_download(w)
        ckpt = torch.load(w)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval()) 

    if len(model) == 1:
        return model[-1]  # return model
    else:
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model 
model=attempt_load("/home/asish/solar/yolov5/runs/train/yolov5s_results2/weights/best.pt")

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

