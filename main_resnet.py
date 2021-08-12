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
from model.ResNet18 import ResNet18
from model.Alexnet import AlexNet
from dataloader.ImageNet32Loader import ImageNet32
# import multiprocessing
import torch.multiprocessing as mp

"""
该联邦学习的思路如下，
首先每个客户端训练epochs个epoch,然后聚合到global model,global model再返回给

"""

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def test(round,model, device, test_loader):
    model.eval()

    correct = 0
    total = 0
    correcttop5 = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            maxk = max((1, 5))
            y_labels = labels.view(-1, 1)
            _, pred = outputs.topk(maxk, 1, True, True)
            correcttop5 += torch.eq(pred, y_labels).sum().float().item()

        time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time_now + ',round '+str(round) +',val Accuracy Top1: %.6f %%' % (100 * correct / float(total)))
        print(time_now + ',round '+str(round) +',val Accuracy Top5: %.6f %%' % (100 * correcttop5 / float(total)))


def client_update(clientId,round,client_model, optimizer, train_loader, epoch,device):
    """
    This function updates/trains client model on client data
    客户端更新
    """
    print("client_update start")

    log_list=[]
    loss_ave=0
    cnt=0
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    client_model.train()
    for e in range(epoch):
        iteration = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            loss_ave+=loss.item()
            cnt+=1

            iteration+=1
            if(iteration%10==0):
                log_str="client id="+str(clientId)+",round="+str(round)+",epoch="+str(e+1)+",client update iteration="+str(iteration)+",loss="+str(loss.item())
                log_list.append(log_str)
                print("client id=",clientId,",round=",round,",epoch=",e+1,",client update iteration=",iteration,",loss=",loss.item())

    return loss_ave/cnt,log_list


def server_aggregate(global_model, client_models):
    """
    This function has aggregation method 'mean'
    服务器端聚合，均值聚合
    """
    ##也需要加噪

    start_add_noise_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("start add noise", start_add_noise_time)

    for model in client_models:
        add_noise_Resnet18(model, args.type, args.batch_size, args.clip_bound, args.pb, args.privacy_delta, args.sample_num,
              param_size, param_size_all, args.is_ratio_list,args.ratio_type)


    end_add_noise_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("end add noise time",end_add_noise_time)


    ### This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)


def send_model(global_model, client_models):

    for model in client_models:
        model.load_state_dict(global_model.state_dict())


if __name__ ==  '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--type',type=str,default='none') #'norm1','norm2','sample_L1','sample_L2','none'
    parser.add_argument('--pb',type=float,default=5e5)
    parser.add_argument('--clip_bound',type=float,default=5)
    parser.add_argument('--log_path',type=str,default="log")
    parser.add_argument('--train_data',type=str,default='data/Imagenet32_train_npz')
    parser.add_argument('--test_data',type=str,default='data/Imagenet32_val_npz')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--test_batch_size',type=int,default=64)
    parser.add_argument('--lr',type=float,default=0.0002)
    parser.add_argument('--momentum',type=float,default=0.5)
    parser.add_argument('-no_cuda',type=str2bool,default=False)
    parser.add_argument('--seed',type=float,default=1)
    parser.add_argument('--log_interval',type=int,default=10)
    parser.add_argument('--privacy_delta',type=float,default=1e-6)
    parser.add_argument('--sample_num',type=int,default=500)
    parser.add_argument('--islog',type=str2bool,default=True)
    parser.add_argument('--is_ratio_list',type=str2bool,default=True)#是否进行自适应的epsilon分配
    parser.add_argument('--num_clients',type=int,default=2)
    parser.add_argument('--num_rounds',type=int,default=10)
    parser.add_argument('--epochs',type=int,default=1)

    #ratio_type控制eps按照权重分配的方法，这里0为权重越大，eps越大，噪声越小
    #1为权重越大，eps越小，噪声越大
    parser.add_argument('--ratio_type',type=int,default=0)

    args = parser.parse_args()


    if args.islog:
        log_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 控制台输出log重定向
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        log_name=log_time+',Resnet'+',type='+args.type+',pb='+str(args.pb) +'.batch_size='+str(args.batch_size) +'.log'
        sys.stdout = Logger(args.log_path+'/'+log_name, sys.stdout)
        sys.stderr = Logger(args.log_path+'/'+log_name, sys.stderr)

    for arg in vars(args):
         print(arg, getattr(args, arg))


    if(args.is_ratio_list):
        print("按照权重分配eps")
        if(args.ratio_type==0):
            print("权重越大,eps越大,噪声越小")
        elif(args.ratio_type==1):
            print("权重越大,eps越小,噪声越大")
    else:
        print("平均分配eps")



    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device",device)

    print("loading data")

    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #训练数据
    train_dataset = ImageNet32(root=args.train_data, train=True, transform=data_tf)
    # 均分训练集,除不尽的补在第一个里面
    length_list=[int(train_dataset.data.shape[0] / args.num_clients) for _ in range(args.num_clients)]
    total_length=0
    total_length=sum(length_list)
    if(total_length!=train_dataset.data.shape[0]):
        length_list[0]+=(train_dataset.data.shape[0]-total_length)

    for length in length_list:
        print(length)

    traindata_split = torch.utils.data.random_split(train_dataset,length_list )

    # Creating a pytorch loader for a Deep Learning model
    train_loader = [torch.utils.data.DataLoader(x, batch_size=args.batch_size, shuffle=True) for x in traindata_split]
    #测试数据
    test_dataset = ImageNet32(root=args.test_data,train=False,transform=data_tf)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.test_batch_size, shuffle=False)

    # #训练数据
    # train_dataset = datasets.CIFAR10(root='./data', train=True, transform=data_tf, download=False)
    # # 均分训练集
    # traindata_split = torch.utils.data.random_split(train_dataset, [int(train_dataset.data.shape[0] / args.num_clients) for _ in range(args.num_clients)])
    # # Creating a pytorch loader for a Deep Learning model
    # train_loader = [torch.utils.data.DataLoader(x, batch_size=args.batch_size, shuffle=True) for x in traindata_split]
    #
    #
    # #测试数据
    # test_dataset = datasets.CIFAR10(root='./data', train=False, transform=data_tf)
    # test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.test_batch_size, shuffle=False)

    print("load data done")

    global_model =  ResNet18().to(device)
    print(global_model)

    client_models = [ResNet18().to(device) for _ in range(args.num_clients)]

    for model in client_models:
        model.load_state_dict(global_model.state_dict())

    opt = [optim.Adam(params=model.parameters(),lr=args.lr) for model in client_models]



    param_size=get_model_param_size(global_model)
    param_size_all=sum(param_size)

    # 训练
    ###### List containing info about learning #########

    # Runnining FL

    print("********start training********")

    for r in range(args.num_rounds):
        # select random clients,这里改成了选择所有的clients
        # client_idx = np.random.permutation(num_clients)[:num_selected]
        client_idx = range(args.num_clients)
        # client update
        loss = 0
        # for i in range(args.num_clients):
        #     print("******client id ",i,"******")
        #     loss += client_update(i,r+1,client_models[i], opt[i], train_loader[client_idx[i]], args.epochs)

        #并行执行
        res=[]
        client_log=[]
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(2)
        for i in range(args.num_clients):
            res.append(pool.apply_async(client_update,(i,r+1,client_models[i], opt[i], train_loader[client_idx[i]], args.epochs,device)))
        pool.close()
        pool.join()

        for i in range(len(res)):
            loss += res[i].get()[0]
            client_log.append(res[i].get()[1])

        for i in range(len(client_log)):
            for line in client_log[i]:
                print(line)

        # server aggregate
        server_aggregate(global_model, client_models)

        print("average train loss=",loss/args.num_clients)
        test(r+1,global_model,device,test_loader)

        send_model(global_model, client_models)


