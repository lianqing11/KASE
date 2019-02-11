from torch.utils.data import DataLoader, Dataset
import numpy as np
import io
from PIL import Image
import pandas as pd
import os.path as osp


class NormalDataset(Dataset):
    def __init__(self, root_dir, meta_file, is_train=True, args=None, transform=None):
        self.root_dir = root_dir
        if not self.root_dir.startswith("/data"):
            self.root_dir = osp.join(osp.expanduser('~'), self.root_dir)
        self.transform = transform
        print("building dataset from %s"%meta_file)
        self.metas = pd.read_csv(meta_file, sep=" ",header=None)
        print("read meta done")
        if is_train==True:
            total_num = args.max_iter*args.batch_size
            self.metas = self.metas.sample(total_num, replace=True)
            self.metas = self.metas.reset_index()
        self.num = len(self.metas)
    def __len__(self):
        return self.num


    def __getitem__(self, idx):
        filename = osp.join(self.root_dir, self.metas.ix[idx, 0])

        label = self.metas.ix[idx, 1]
        ## memcached
        img = Image.open(filename).convert('RGB')
        #img = np.zeros((350, 350, 3), dtype=np.uint8)
        #img = Image.fromarray(img)
        #cls = 0

        ## transform
        if self.transform is not None:
            img = self.transform(img)
        return img, label




class TeacherDataset(Dataset):
    def __init__(self, root_dir, meta_file, transform=None, args=None, is_train=True):
        self.root_dir = root_dir
        if not self.root_dir.startswith("/data"):
            self.root_dir = osp.join(osp.expanduser('~'), self.root_dir)
        self.transform = transform
        metas = pd.read_csv(meta_file, sep=" ", header=None)
        if args != None and is_train==True:
            metas = metas.sample(args.max_iter*args.batch_size, replace=True)
            metas = metas.reset_index()
        self.metas = metas
        print("building dataset from %s"%meta_file)
        self.num = len(self.metas)

        print("read meta done")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename = self.root_dir + '/' + self.metas.ix[idx,0]
        cls = self.metas.ix[idx,1]

        ## memcached

        img = Image.open(filename).convert('RGB')

        ## transform
        if self.transform is not None:
            img1, img2 = self.transform(img)

        return img1, img2, cls
