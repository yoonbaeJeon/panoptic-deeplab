#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pickle as pkl
import sys
import torch

"""
Usage:
  # download your pretrained model:
  wget https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/tf-xception65-270e81cf.pth -O x65.pth
  # run the conversion
  ./convert-pretrained-model-to-d2.py x65.pth x65.pkl
  # Then, use x65.pkl with the following changes in config:

MODEL:
  WEIGHTS: "/path/to/x65.pkl"
  PIXEL_MEAN: [128, 128, 128]
  PIXEL_STD: [128, 128, 128]
INPUT:
  FORMAT: "RGB"
"""

if __name__ == "__main__":
    input = sys.argv[1]

    state_dict = torch.load(input, map_location="cpu")
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'module' in state_dict:
        state_dict = state_dict['module']
    res = {"model": state_dict, "__author__": "third_party", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
