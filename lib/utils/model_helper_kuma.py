import os
import glob
import math
import torch
import torch.nn as nn
from os import makedirs
from copy import deepcopy
from os.path import join, exists
from loguru import logger
from utils.general_helper import is_parallel

def is_valid_number(x):
    return not(math.isnan(x) or math.isinf(x) or x > 1e4)