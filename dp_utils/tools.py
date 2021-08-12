
import torch
import numpy as np
import random
import math
import sys
from datetime import datetime
#梯度裁剪功能
def clip_func(clip_bound,clip_type,input):
    if(clip_bound<=0):
        return input
    if(clip_type=="norm1"):
        return torch.clamp(input,-1*clip_bound,clip_bound)
    elif(clip_type=="norm2"):
        norm2=torch.norm(input).item()
        tmp=max(norm2/clip_bound,1)
        return input/tmp
    else:
        print("no such clip-type")
        return input


#拉普拉斯噪声,高斯噪声
def laplace_function(beta,size):
    return np.random.laplace(0,beta,size=size)


def gauss_function(sigma):
    return random.gauss(0,sigma)

def get_tensor_size(param_size):
    tmp=1
    for i in param_size:
        tmp*=i
    return tmp


#计算所有参数梯度的敏感度
def calculate_l1_sensitivity(clip_bound,param_size):
    return 2*clip_bound*param_size

def calculate_l2_sensitivity(clip_bound):
    return 2*clip_bound

def calculate_l1_sensitivity_sample(grad_data,param_size,sample_num):
    # sample
    grad_data_1D=grad_data.view(param_size)
    if(sample_num<=param_size):
        sample_index=random.sample(range(param_size),sample_num)
        sample_grad = grad_data_1D[sample_index]
    else:
        sample_grad=grad_data_1D

    #print(sample_gard)
    #计算标准差
    std_deviation=sample_grad.std().item()
    return (1.13*param_size+2.56*math.sqrt(param_size))*std_deviation


def calculate_l2_sensitivity_sample(grad_data,param_size,sample_num):
    # sample
    grad_data_1D=grad_data.view(param_size)
    if(sample_num<=param_size):
        sample_index=random.sample(range(param_size),sample_num)
        sample_grad = grad_data_1D[sample_index]
    else:
        sample_grad=grad_data_1D
    #print(sample_gard)
    #计算标准差
    std_deviation=sample_grad.std().item()
    return 1.45*math.sqrt(param_size)*std_deviation


def gen_laplace_beta(batchsize,Parallelnum,sensitivity,privacy_budget):

    scaledEpsilon=privacy_budget*float(batchsize)/Parallelnum
    beta=sensitivity/scaledEpsilon
    return beta

def gen_gaussian_sigma(batchsize,Parallelnum,sensitivity,privacy_budget,privacyDelta):
    scaledEpsilon = privacy_budget * float(batchsize) / Parallelnum
    sigma= (2.0 * math.log(1.25 / privacyDelta)) * sensitivity / scaledEpsilon
    return sigma

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_param_size(model):
    param_size=[]
    for param in model.parameters():
        param_size.append(get_tensor_size(param.data.size()))
    return  param_size


# log
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def print_ratio_list(param_data_ratio_list):
    for index, value in enumerate(param_data_ratio_list):
        print("layer=", index, "ratio=", value)



def conver_ratio_list(ratio_list): #把ratio list转化成和为1的list
    list_sum=sum(ratio_list)
    for i in range(len(ratio_list)):
        ratio_list[i]/=list_sum
    return ratio_list


def add_noise(model,type,batch_size,clip_bound,privacy_budget,privacy_delta,sample_num,param_sizes,param_size_all,is_ratio_list,ratio_type):

    if type == "none":
        return


    with torch.no_grad():

        ## 如果是norm1 norm2方法，需要对parm.data进行clip,clip之后才能计算ratio
        if type == "norm1" or type=="norm2":
            for parm in model.parameters():
                parm.data = clip_func(clip_bound, type, parm.data)

        param_data_ratio_list = []

        if is_ratio_list:

            # ratio_type为0的情况，权重越大eps越大
            for parm, parm_size in zip(model.parameters(), param_sizes):
                total=torch.sum(torch.abs(parm)).item()
                param_data_ratio_list.append(total/parm_size)

            if ratio_type == 1:  # 权重越大，eps越小，噪声越大
                param_data_ratio_list = conver_ratio_list(param_data_ratio_list)
                for i in range(len(param_data_ratio_list)):
                    param_data_ratio_list[i] = 1.0 / param_data_ratio_list[i]  # 改成反比
            # 还有一个优化,层数越深，加噪越小,Yosinski, Jason, et al."How transferable are features in deep neural networks?."Advances in neural information processing systems.2014.
            elif ratio_type==2: ##深度越深，eps越大
                for i in range(len(param_data_ratio_list)):
                    if i==0:
                        param_data_ratio_list[i]=1.0
                    else:
                        param_data_ratio_list[i]=param_data_ratio_list[i-1]*1.2


        else:
            for parm_size in param_sizes:
                param_data_ratio_list.append(1)

        param_data_ratio_list = conver_ratio_list(param_data_ratio_list)
        print_ratio_list(param_data_ratio_list)



        cnt=0
        if type == "norm1":
            for parm in model.parameters():
                sensitivity = calculate_l1_sensitivity(clip_bound, param_size_all)
                beta = gen_laplace_beta(batch_size, 1, sensitivity, privacy_budget*param_data_ratio_list[cnt])
                noise_tensor = torch.from_numpy(laplace_function(beta, parm.data.size())).float()
                parm.data += noise_tensor.cuda()
                cnt+=1



        elif type == "norm2":
            for parm in model.parameters():
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



def add_noise_Resnet18(model,type,batch_size,clip_bound,privacy_budget,privacy_delta,sample_num,param_sizes,param_size_all,is_ratio_list,ratio_type):


    if type == "none":
        return

    with torch.no_grad():

        ## 如果是norm1 norm2方法，需要对parm.data进行clip,clip之后才能计算ratio
        if type == "norm1" or type=="norm2":
            for parm in model.parameters():
                parm.data = clip_func(clip_bound, type, parm.data)


        '''
        按照bolck添加
        共分为conv1,layer1,layer2,layer3,layer4,fc
             conv1 [0-2] 3
             layer1 12
             layer2 15
             layer3 15
             layer4 15
             fc 2
        '''
        layer_map = []  # 层到block的一个映射
        for i in range(3):
            layer_map.append(0)
        for i in range(12):
            layer_map.append(1)
        for i in range(15):
            layer_map.append(2)
        for i in range(15):
            layer_map.append(3)
        for i in range(15):
            layer_map.append(4)
        for i in range(2):
            layer_map.append(5)

        param_data_ratio_list=[]     #62层每一层的weight
        layer_ratio_list=[0,0,0,0,0,0]   #记录conv1 layer1,layer2...的weight
        layer_parm_size=[0,0,0,0,0,0]

        #使用按权重分配
        if is_ratio_list:

            # ------------------------ratio_type为0的情况，权重越大eps越大------------------------
            cnt=0
            for parm, parm_size in zip(model.parameters(), param_sizes):
                total=torch.sum(torch.abs(parm)).item()
                layer_parm_size[layer_map[cnt]]+=parm_size #记录conv1 layer1,layer2...对应的size
                layer_ratio_list[layer_map[cnt]]+=total #记录conv1 layer1,layer2...对应的绝对值总大小
                cnt+=1

            for index,val in enumerate(layer_ratio_list):
                layer_ratio_list[index]/=layer_parm_size[index]  #求一下平均值，layer_ratio_list中存储了conv1 layer1,layer2...对应的参数绝对值平均数

            for i in range(62):
                param_data_ratio_list.append(layer_ratio_list[layer_map[i]])   #按照分布把layer_ratio_list换成param_data_ratio_list


            #------------------------ratio_type为1的情况,权重越大，eps越小，噪声越大------------------------
            if ratio_type == 1:
                param_data_ratio_list = conver_ratio_list(param_data_ratio_list)
                for i in range(len(param_data_ratio_list)):
                    param_data_ratio_list[i] = 1.0 / param_data_ratio_list[i]  # 改成反比

            #------------------------ratio_type为2,层数越深，加噪越小------------------------
            # Yosinski, Jason, et al."How transferable are features in deep neural networks?."Advances in neural information processing systems.2014.

            elif ratio_type ==2: ##深度越深，eps越大
                for i in range(len(layer_ratio_list)):
                    if i==0:
                        layer_ratio_list[i]=1.0
                    else:
                        layer_ratio_list[i]=layer_ratio_list[i-1]*1.5  #逐级加大eps的ratio,1.5的系数可以更改

                for i in range(62):
                    param_data_ratio_list[i]=layer_ratio_list[layer_map[i]] # 按照分布把layer_ratio_list换成param_data_ratio_list

        else:
            for parm_size in param_sizes:
                param_data_ratio_list.append(1)


        param_data_ratio_list = conver_ratio_list(param_data_ratio_list)
        print_ratio_list(param_data_ratio_list)


        cnt=0
        if type == "norm1":
            for parm in model.parameters():
                sensitivity = calculate_l1_sensitivity(clip_bound, param_size_all)
                beta = gen_laplace_beta(batch_size, 1, sensitivity, privacy_budget*param_data_ratio_list[cnt])
                noise_tensor = torch.from_numpy(laplace_function(beta, parm.data.size())).float()
                parm.data += noise_tensor.cuda()
                cnt+=1



        elif type == "norm2":
            for parm in model.parameters():
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


def add_noise_test(model,type,batch_size,clip_bound,privacy_budget,privacy_delta,sample_num,param_sizes,param_size_all,is_ratio_list,ratio_type,test_layer):

    #这个作为测试，只为某一层添加噪声
    print("***************************add noise test***************************")
    print("test layer=",test_layer)
    if type == "none":
        return


    with torch.no_grad():

        ## 如果是norm1 norm2方法，需要对parm.data进行clip,clip之后才能计算ratio
        if type == "norm1" or type=="norm2":
            for parm in model.parameters():
                parm.data = clip_func(clip_bound, type, parm.data)

        cnt=0
        if type == "norm1":
            for parm in model.parameters():
                if cnt==test_layer:
                    sensitivity = calculate_l1_sensitivity(clip_bound, param_size_all)
                    beta = gen_laplace_beta(batch_size, 1, sensitivity, privacy_budget)
                    noise_tensor = torch.from_numpy(laplace_function(beta, parm.data.size())).float()
                    parm.data += noise_tensor.cuda()
                cnt+=1



        elif type == "norm2":
            for parm in model.parameters():
                if cnt==test_layer:
                    sensitivity = calculate_l2_sensitivity(clip_bound)
                    sigma = gen_gaussian_sigma(batch_size, 1, sensitivity, privacy_budget, privacy_delta)
                    noise_tensor = torch.normal(torch.zeros(parm.data.size()), sigma)
                    print("noise tensor maxVal=",torch.max(noise_tensor))
                    print("noise tensor minVal=", torch.min(noise_tensor))
                    parm.data += noise_tensor.cuda()
                cnt+=1

        elif type == "sample_L1":
            for parm in model.parameters():
                if cnt==test_layer:
                    tensor_size = param_sizes[cnt]
                    sensitivity = calculate_l1_sensitivity_sample(parm.data, tensor_size, sample_num)*1.35   #采样数为500时，需要对标准差乘以1.35的修正系数
                    beta = gen_laplace_beta(batch_size, 1, sensitivity, privacy_budget)
                    noise_tensor = torch.from_numpy(laplace_function(beta, parm.data.size())).float()
                    parm.data += noise_tensor.cuda()
                cnt+=1

        elif type == "sample_L2":
            for parm in model.parameters():
                if cnt==test_layer:
                    tensor_size = param_sizes[cnt]
                    sensitivity = calculate_l2_sensitivity_sample(parm.data, tensor_size, sample_num)*1.35
                    sigma = gen_gaussian_sigma(batch_size, 1, sensitivity, privacy_budget, privacy_delta)
                    noise_tensor = torch.normal(torch.zeros(parm.data.size()), sigma)
                    parm.data += noise_tensor.cuda()
                cnt+=1

