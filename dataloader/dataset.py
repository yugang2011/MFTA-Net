import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class TrainSet(Dataset):
    def __init__(self, args):
        self.input_list = sorted([os.path.join(args.train_data_lr, name) for name in os.listdir(args.train_data_lr)])
        self.ref_list = sorted([os.path.join(args.train_data_ref, name) for name in os.listdir(args.train_data_ref)])
        self.hr_list = sorted([os.path.join(args.train_data_hr, name) for name in os.listdir(args.train_data_hr)])
        self.transform = transforms.ToTensor()
        self.crop_size = args.crop_size

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        HR = np.load(self.hr_list[idx])
        LR = np.load(self.input_list[idx])
        Ref = np.load(self.ref_list[idx])

        HRL3 = HR.resize((64, 64), Image.BICUBIC)
        HRL2 = HR.resize((128, 128), Image.BICUBIC)

        RefL3 = Ref.resize((64, 64), Image.BICUBIC)
        RefL2 = Ref.resize((128, 128), Image.BICUBIC)

        sample = {'HR': HR,
                  'LR': LR,
                  'Ref':Ref,
                  'HR2': HRL2,
                  'HR3': HRL3,
                  'Ref2': RefL2,
                  'Ref3': RefL3,
                  }
        for key in sample.keys():
            sample[key] = transforms.ToTensor()(sample[key]).float()
        return sample


class TestSet(Dataset):
    def __init__(self, args):
        self.input_list = sorted([os.path.join(args.test_data_lr, name) for name in os.listdir(args.test_data_lr)])
        self.ref_list = sorted([os.path.join(args.test_data_ref, name) for name in os.listdir(args.test_data_ref)])
        self.hr_list = sorted([os.path.join(args.test_data_hr, name) for name in os.listdir(args.test_data_hr)])
        self.scale = args.sr_scale
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        HR_name = os.path.basename(self.input_list[idx])

        HR = np.load(self.hr_list[idx])
        Ref = np.load(self.ref_list[idx])
        LR = np.load(self.input_list[idx])

        sample = {'HR': HR,
                  'LR': LR,
                  'Ref':Ref
                  }
        for key in sample.keys():

            sample[key] = transforms.ToTensor()(sample[key]).float()

        sample['HR_name'] = HR_name+'.png'

        return sample

