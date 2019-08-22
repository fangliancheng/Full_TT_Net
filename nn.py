from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import svd_dataset
import time

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        before, after = sample['before'], sample['after']
        #print('before',before.as_matrix())
        #print('type',type(before.as_matrix()))
        return {'before': torch.from_numpy(before.as_matrix()),
                'after': torch.from_numpy(after.as_matrix())}


class LeNet_300_100(nn.Module):

    def __init__(self):
        super(LeNet_300_100, self).__init__()

        self.ip1 = nn.Linear(25+5+25,300)
        self.relu_ip1 = nn.ReLU(inplace=True)
        self.ip2 = nn.Linear(300,100)
        self.relu_ip2 = nn.ReLU(inplace=True)
        self.ip3 = nn.Linear(100,100)
        self.relu_ip3 = nn.ReLU(inplace=True)
        self.ip4 = nn.Linear(100,55)

    def forward(self, x):
        x = x.view(-1,55)
        x = self.ip1(x)
        x = self.relu_ip1(x)
        x = self.ip2(x)
        x = self.relu_ip2(x)
        x = self.ip3(x)
        x = self.relu_ip3(x)
        x = self.ip4(x)
        return x

#TODO:training function
def train(args,model,device,train_loader,optimizer,epoch):
    model.train()
    print(model)
    for batch_idx,sample_batched in enumerate(train_loader):
        data,target = sample_batched['before'].to(device), sample_batched['after'].to(device)
        #print('data_size',data.size())
        #print('target_size', target.size())
        optimizer.zero_grad()
        output = model(data.float())
        cri = nn.MSELoss(reduction='mean')
        loss = cri(output,target.float())
        loss.backward()
        #gradient clipping
        #torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        optimizer.step()
        #running_loss = 0.0
        #running_loss += loss.item()
        #print('loss.item():', loss.item())
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

#TODO:adjust_learning_rate function
#or reconstruct a optimizer with different lr
#
def ground_truth(u,s,v):
    usv = torch.mm(torch.mm(u,torch.diag(s)),v.t())
    for i in range(5):
        for j in range(5):
            if usv[i,j] < 0:
                usv[i,j] = 0
    return usv

def forward_time_compare(model):
    transformed_dataset = svd_dataset.svd_dataset(csv_file='svd_data.csv',transform=ToTensor())
    sample = transformed_dataset[32]
    data = sample['before']
    start_time = time.time()
    output = model(data.float())
    print('NN forward pass time:',time.time() - start_time)
    #reconstruct full format using u,s,v, then do ReLU elementwise, then decompose by svd again
    a = 20*torch.randn(5,5)
    u,s,v = torch.svd(a,some=False)
    while s.size()[0]<5:
        print('work')
        s = torch.from_numpy(np.pad(s.numpy(),(0,1),'constant'))
    start_time2 = time.time()
    #reconstuct plus ReLU
    temp = ground_truth(u,s,v)
    u,s,v = torch.svd(a,some=False)
    print('baseline running time',time.time() - start_time2)
    return



#TODO:main function
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--clip',type=float, default=0.25,help='gradient clipping')
    parser.add_argument('--iteration', type=int, default=20000, metavar='N',
                                                        help='iteration steps for training (default: 20)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                        help='learning rate (default: 1e-6)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda:', use_cuda)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset = svd_dataset.svd_dataset(csv_file='svd_data.csv',transform=ToTensor())
    train_loader = torch.utils.data.DataLoader(
                   dataset, batch_size=args.batch_size,
                   shuffle=True, **kwargs)

    model = LeNet_300_100().to(device)
    forward_time_compare(model)
    lr = args.lr
    optimizer_1 = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    optimizer_2 = optim.SGD(model.parameters(),lr=lr/3,momentum=args.momentum)
    optimizer_3 = optim.SGD(model.parameters(),lr=lr/10,momentum=args.momentum)

    #
    # a = 20*torch.randn(5,5)
    # print('a:',a)
    # u,s,v = torch.svd(a,some=False)
    # print('u:',u)
    # print('s',s)
    # print('v',v)
    # while s.size()[0]<5:
    #     print('work')
    #     s = torch.from_numpy(np.pad(s.numpy(),(0,1),'constant'))
    #
    # input = torch.cat((u.reshape(25,-1),s.reshape(5,-1),v.reshape(25,-1)),0).to(device)
    # print('input:',input)
    # print('input init shape:', input.size())
    # #squeeze last dimension, same for target
    # input = torch.squeeze(input).to(device)
    # #add first batch dimension
    # input = torch.unsqueeze(input,0).to(device)
    # print('input size:', input.size())
    #
    # #same for target
    # target = ground_truth(u,s,v).reshape(25,-1).to(device)
    # print('target:',target)
    # print('target init shapeL:', target.size())
    # target = torch.squeeze(target).to(device)
    # target = torch.unsqueeze(target,0).to(device)
    # print('target size:', target.size())
    #
    # print('gt:', ground_truth(u,s,v))
    # print('nn output init:', model(input), model(input).size())
    # train(input, target, args, model, device, optimizer_1, args.iteration, epoch=1)
    # print('nn output ep1:', model(input))
    # train(input, target, args, model, device, optimizer_2, args.iteration, epoch=2)
    # print('nn output ep2:', model(input))
    # train(input, target, args, model, device, optimizer_3, args.iteration, epoch=3)
    # print('nn output ep3:', model(input))

    for epoch in range(1,args.epochs + 1):
        train(args,model,device,train_loader,optimizer_1,epoch)

if __name__ == '__main__':
        main()
