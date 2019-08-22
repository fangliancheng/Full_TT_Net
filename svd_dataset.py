import torch
from torch.utils import data
import pandas as pd
from torchvision import transforms

class svd_dataset(data.Dataset):
    def __init__(self, csv_file,transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        before = self.data.iloc[index,:55]
        after = self.data.iloc[index,55:]
        sample = {'before':before,'after':after}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        before, after = sample['before'], sample['after']
        print('before',before.as_matrix())
        print('type',type(before.as_matrix()))
        return {'before': torch.from_numpy(before.as_matrix()),
                'after': torch.from_numpy(after.as_matrix())}

#test
if __name__ == '__main__':
    transformed_dataset = svd_dataset(csv_file='svd_data.csv',transform=ToTensor())
    #transformed_dataset = svd_dataset(csv_file='svd_data.csv',transform=transforms.Compose([ToTensor()]))

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print('sample',sample)
        print(i, sample['before'].size(), sample['after'].size())

        if i == 3:
            break
