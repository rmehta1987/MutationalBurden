import pandas as pd
import re   
import torch
from torch.utils.data import Dataset
import numpy as np

class CustomCSVDataset(Dataset):
    def __init__(self, filenames, batch_size, max_zero_pad):
        '''Initializes dataset params
        Args:
        filenames (_type_): is a list of strings that contains all file names.
        batch_size (_type_):  determines the number of files that we want to read in a chunk.
        ''' 
        self.filenames= filenames
        self.batch_size = batch_size
        self.max_pad = max_zero_pad

    def __len__(self):

        return int(np.ceil(len(self.filenames) / float(self.batch_size)))   # Number of chunks.
    
    def __getitem__(self, idx):
        ''' 
        # In this method, we do all the preprocessing.
        # First read data from files in a chunk. Preprocess it. Extract labels. Then return data and labels.
        Args:
        idx (_type_): idx means index of the chunk.
        Raises:
        IndexError: if index is at the last index of the file

        Returns:
        _type_: locus, estimated beta, standard error, masked-array
        '''
        if len(self.filenames) == 1:
            batch_x = self.filenames
        else:
            batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]   # This extracts one batch of file names from the list `filenames`.
        beta = []
        locus = []
        standard_error = []
        mask = []
        file_names = []
        for file in batch_x:
            fo = open(file,'r')
            temp = pd.read_csv(fo,delimiter='\t') # Change this line to read any other type of file
            temp.columns = temp.columns.str.lower()
            assert ['locus', 'beta', 'se'] == list(temp.columns), print("Columns are not correct in file, should be locus, beta, se (standard error), current columns are {}".format(temp.columns))
            beta_temp = list(temp['beta'])
            standard_error_temp = list(temp['se'])
            beta_tensor = torch.nn.functional.pad(torch.tensor(beta_temp,dtype=torch.float32),pad=(0,self.max_pad-len(beta_temp)),value=0)
            standard_error_tensor = torch.nn.functional.pad(torch.tensor(standard_error_temp,dtype=torch.float32),pad=(0,self.max_pad-len(standard_error_temp)),value=0)
            mask_tensor = (~(beta_tensor == 0)).float() # Variants where actually exist are 1, everything else is 0
            beta.append(beta_tensor)
            locus.append(temp['locus'])
            mask.append(mask_tensor)
            standard_error.append(standard_error_tensor)
            file_name = file.split('/')[-1][:-3]
            file_names.append(file_name)
            fo.close()


        # The following condition is actually needed in Pytorch. Otherwise, for our particular example, the iterator will be an infinite loop.
        # Readers can verify this by removing this condition.
        if idx == self.__len__():  
            raise IndexError

        return {'beta': torch.stack(beta,dim=0), 'mask': torch.stack(mask,dim=0), 'locus': locus, 'se': torch.stack(standard_error,dim=0), 'gene': file_names}