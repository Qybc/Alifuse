import os
from glob import glob
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import SimpleITK as sitk


label_map = {
    'AD':2,
    'Dementia':2,
    'Demented':2,
    'MCI':1,
    'CN':0,
    'Control':0,
    'Nondemented':0,
}

class Dataset(torch.utils.data.Dataset):
  """
  Loads data and corresponding label and returns pytorch float tensor.
  """
  def __init__(self, data):
    self.files = data

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    """
    Read data and label and return them.
    """
    img_path = self.files[idx]['image_path']
    data = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)
    data = torch.FloatTensor(data)
    # 归一化到[0,1]
    data = F.normalize(data)
    # pad成为cube
    shape = max(data.shape)
    data = F.pad(data,[0,shape-data.shape[0],0,shape-data.shape[1],0,shape-data.shape[2]])
    # resize
    image = F.interpolate(data.unsqueeze(0).unsqueeze(0),size=(128,128,128),mode='trilinear').squeeze()

    text = self.files[idx]['text']
    label = self.files[idx]['label']

    

    return image, text, label


def get_dataloader(datalist=['ADNI-train'],batch_size=1,shuffle=False,num_workers=12, drop_last=False):
    files = []
    for data in datalist:
        filename = f'./local_data/{data}.csv'
        print('load data from', filename)
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                imgpath,report = line.strip('\n').split('\t')
                if 'The diagnosis is' in report:
                    text = report.split('The diagnosis is ')[0]
                    name = report.split('The diagnosis is ')[1].split('.')[0]
                    if name in label_map.keys():
                        label = label_map[name]
                    else:
                        label=-100
                else:
                    text = report
                    label = -100

                files.append(
                    {
                        'image_path': imgpath,
                        'text': text,
                        'label': label
                    }
                )
    dataset = Dataset(data=files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,drop_last=drop_last)

    return dataloader