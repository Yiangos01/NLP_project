
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.functional as F
import numpy as np
# from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


dataset = torch.load('data/demo.valid.pt')
file = open("test_src_preprocesed.txt","w") 
file2 = open("test_tgt_preprocesed.txt","w") 


for i in range(len(dataset)):
	src= dataset[i].src
	src= ' '.join(src)
	tgt= dataset[i].tgt
	tgt= ' '.join(tgt)
	src=normalizeString(src)
	tgt=normalizeString(tgt)
	print(src,tgt)
	file2.write(tgt+"\n")
	file.write(src+"\n") 
