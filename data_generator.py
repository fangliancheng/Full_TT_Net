 import torch
import pandas as pd
import csv
import numpy as np

def create_csv(csv_head):
    path = "svd_data.csv"
    with open(path,'w') as f:
        csv_write = csv.writer(f)
        #csv_head = ["in_entity","out_entity"]
        csv_write.writerow(csv_head)

#data_row: ["1","2"]
def write_csv(data_row):
    path = "svd_data.csv"
    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)

#return is a matrix, need to de svd again
def ground_truth(u,s,v):
    usv = torch.mm(torch.mm(u,torch.diag(s)),v.t())
    for i in range(5):
        for j in range(5):
            if usv[i,j] < 0:
                usv[i,j] = 0
    return usv

#for a 5*5 input matrix, applying full SVD, we get 55 input entities, we create ground_truth output by apply ReLU to ground_truth input
#matrix, then svd its output, so output will also have 55 entities.
if __name__ == '__main__':
    #TODO:done
    #[0,1,2,...,109]
    a = torch.arange(0,110)
    csv_head = [str(element.item()) for element in a.flatten()]
    create_csv(csv_head)
    dataset_len = 10000
    for i in range(1,dataset_len):
        scale = torch.torch.randn(1).float().uniform_(0,1000)
        ori_matrix = scale * torch.randn(5,5)
        u,s,v = torch.svd(ori_matrix,some=False)
        while s.size()[0]<5:
            print('Need auto_fill_zero')
            s = torch.from_numpy(np.pad(s.numpy(),(0,1),'constant'))

        input = torch.cat((u.reshape(25,-1),s.reshape(5,-1),v.reshape(25,-1)),0)
        #squeeze last dimension, same for target
        input = torch.squeeze(input) #might be unnecssary
        input_list = [str(element.item()) for element in input.flatten()]

        target_matrix = ground_truth(u,s,v)
        u_t,s_t,v_t = torch.svd(target_matrix,some=False)
        while s_t.size()[0]<5:
            print('Need auto_fill_zero')
            s_t = torch.from_numpy(np.pad(s_t.numpy(),(0,1),'constant'))
        target = torch.cat((u_t.reshape(25,-1),s_t.reshape(5,-1),v_t.reshape(25,-1)),0)
        target = torch.squeeze(target)
        output_list = [str(element.item()) for element in target.flatten()]

        write_csv(input_list + output_list)
