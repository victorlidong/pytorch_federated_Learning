import torch
import torch .nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
torch.set_default_tensor_type(torch.cuda.FloatTensor) # <-- workaround
import syft as sy
from dp_utils import *
import argparse
from datetime import datetime
import sys
import os

"""
norm1 norm2 能否有一个好的效果与clipbound的选择密切相关，因为这不是梯度，模型参数的大小有着严格的要求


norm2 下，bound设置0.1就怎么也收敛不了

几个结论，在bound选择小的时候，对参数裁剪的影响很大，大于加噪声的影响，即使什么噪声也不加，仅仅裁剪参数，也会导致准确率大幅降低


"""


parser = argparse.ArgumentParser()


parser.add_argument('--type',type=str,default='none') #'norm1','norm2','sample_L1','sample_L2','none'
parser.add_argument('--pb',type=float,default=1e5)
parser.add_argument('--clip_bound',type=float,default=5)
parser.add_argument('--log_path',type=str,default="log")
parser.add_argument('--epochs',type=int,default=10)
parser.add_argument('--train_data',type=str,default='data')
parser.add_argument('--test_data',type=str,default='data')
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--test_batch_size',type=int,default=1000)
parser.add_argument('--lr',type=float,default=0.01)
parser.add_argument('--momentum',type=float,default=0.5)
parser.add_argument('--no_cuda',type=bool,default=False)
parser.add_argument('--seed',type=float,default=1)
parser.add_argument('--log_interval',type=int,default=10)
parser.add_argument('--privacy_delta',type=float,default=1e-6)
parser.add_argument('--sample_num',type=int,default=500)
parser.add_argument('--islog',type=bool,default=True)
parser.add_argument('--is_ratio_list',type=bool,default=True)#是否进行自适应的epsilon分配
args = parser.parse_args()





if args.islog:
    log_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 控制台输出log重定向
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    log_name=log_time+',type='+args.type+',pb='+str(args.pb) +'.batch_size='+str(args.batch_size) +'.log'
    sys.stdout = Logger(args.log_path+'/'+log_name, sys.stdout)
    sys.stderr = Logger(args.log_path+'/'+log_name, sys.stderr)


for arg in vars(args):
     print(arg, getattr(args, arg))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



def add_noise(model,type,batch_size,clip_bound,privacy_budget,privacy_delta,sample_num,param_sizes,param_size_all,is_ratio_list): #待完成
    # for name in model.state_dict(): #某一层的参数名字
    #     print(name)
    #
    # for param in model.parameters():
    #     param.data *=0
    #     # do something to param.data

    with torch.no_grad():

        param_data_ratio_list = []
        if is_ratio_list:
            param_data_average_list=[]
            for parm, parm_size in zip(model.parameters(), param_sizes):
                total=torch.sum(torch.abs(parm)).item()
                param_data_average_list.append(total/parm_size)
            total_sum=sum(param_data_average_list)
            for value in param_data_average_list:
                param_data_ratio_list.append(value/total_sum)
        else:
            for parm_size in param_sizes:
                param_data_ratio_list.append(1.0/len(param_sizes))



        cnt=0
        if type == "norm1":
            for parm in model.parameters():
                parm.data = clip_func(clip_bound, type, parm.data)
                sensitivity = calculate_l1_sensitivity(clip_bound, param_size_all)
                beta = gen_laplace_beta(batch_size, 1, sensitivity, privacy_budget*param_data_ratio_list[cnt])
                noise_tensor = torch.from_numpy(laplace_function(beta, parm.data.size())).float()
                parm.data += noise_tensor.cuda()
                cnt+=1

        elif type == "norm2":
            for parm in model.parameters():
                parm.data = clip_func(clip_bound, type, parm.data)
                sensitivity = calculate_l2_sensitivity(clip_bound)
                sigma = gen_gaussian_sigma(batch_size, 1, sensitivity, privacy_budget*param_data_ratio_list[cnt], privacy_delta)
                noise_tensor = torch.normal(torch.zeros(parm.data.size()), sigma)
                parm.data += noise_tensor.cuda()
                cnt+=1

        elif type == "sample_L1":
            for parm in model.parameters():
                tensor_size = param_sizes[cnt]
                sensitivity = calculate_l1_sensitivity_sample(parm.data, tensor_size, sample_num)*1.35   #采样数为500时，需要对标准差乘以1.35的修正系数
                beta = gen_laplace_beta(batch_size, 1, sensitivity, privacy_budget*param_data_ratio_list[cnt])
                noise_tensor = torch.from_numpy(laplace_function(beta, parm.data.size())).float()
                parm.data += noise_tensor.cuda()
                cnt+=1

        elif type == "sample_L2":
            for parm in model.parameters():
                tensor_size = param_sizes[cnt]
                sensitivity = calculate_l2_sensitivity_sample(parm.data, tensor_size, sample_num)*1.35
                sigma = gen_gaussian_sigma(batch_size, 1, sensitivity, privacy_budget*param_data_ratio_list[cnt], privacy_delta)
                noise_tensor = torch.normal(torch.zeros(parm.data.size()), sigma)
                parm.data += noise_tensor.cuda()
                cnt+=1



def train(model, device, train_loader, optimizer, epoch,param_size,param_size_all):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader): # <-- now it is a distributed dataset

        # 需要为model的参数加噪
        add_noise(model, args.type, args.batch_size, args.clip_bound, args.pb, args.privacy_delta, args.sample_num,param_size,param_size_all,args.is_ratio_list)

        model.send(data.location) # <-- NEW: send the model to the right location
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        model.get() # <-- NEW: get/media/aaa/041CDACD1CDAB93E/pyProject/pytorch_Federated Learning/log/sampleL2 the model back
        if batch_idx % args.log_interval == 0:
            loss = loss.get() # <-- NEW: get the loss back
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader) * args.batch_size, #batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print(data.location.id)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Accuracy:='+str(correct / len(test_loader.dataset)))



use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

print("device",device)


# kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
kwargs={}

hook =sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")


federated_train_loader = sy.FederatedDataLoader( # <-- this is now a FederatedDataLoader
    datasets.MNIST(args.train_data, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    .federate((bob, alice)), # <-- NEW: we distribute the dataset across all the workers, it's now a FederatedDataset
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.test_data, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)



model = Net().to(device)
print(model)

param_size=get_model_param_size(model)
param_size_all=sum(param_size)

optimizer = optim.SGD(model.parameters(), lr=args.lr) # TODO momentum is not supported at the moment

for epoch in range(1, args.epochs + 1):
    train(model, device, federated_train_loader, optimizer, epoch,param_size,param_size_all)
    test( model, device, test_loader)

