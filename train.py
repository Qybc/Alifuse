import os
import random

import numpy as np
import torch
from model.modeling_alifuse import AlifuseModel
from dataset.dataset import get_dataloader
from model.trainer import Trainer

# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM']='false'

# only pretrain on ADNI train data and NACC train data
train_datalist = [
    'ADNI-train',
    'NACC-train',
]

val_datalist = [
    'ADNI-test',
    # 'NACC-test',
    # 'AIBL',
    # 'OASIS2',
    # 'MIRIAD',
]

trainloader = get_dataloader(train_datalist, batch_size=5,shuffle=True,num_workers=2, drop_last=True)
valloader = get_dataloader(val_datalist, batch_size=24,shuffle=False,num_workers=2, drop_last=False)

model = AlifuseModel()
model.cuda()
trainer = Trainer()
trainer.train(
    model,
    trainloader,
    valloader,
    warmup_ratio=0.1,
    epochs=100,
    optimizer_params={'lr':2e-5},
    output_path=f'./checkpoints/litc_lres_lcls/',
    metric_path=f'./checkpoints/litc_lres_lcls_metrics.txt',
    weight_decay=1e-4,
    )
    

