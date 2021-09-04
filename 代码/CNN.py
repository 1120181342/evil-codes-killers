from viper.common.abstracts import Module
from viper.core.session import __sessions__
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from PIL import Image 
from viper.core.database import Database
import numpy as np
import os
import argparse

class CNNMalware_Model1(nn.Module):
    def __init__(self, image_dim=256, num_of_classes=7):
        super().__init__()

        self.image_dim = image_dim
        self.num_of_classes = num_of_classes

        self.conv1_out_channel = 12
        self.conv1_kernel_size = 3

        self.conv2_out_channel = 16
        self.conv2_kernel_size = 3

        self.linear1_out_features = 120
        self.linear2_out_features = 90

        self.conv1 = nn.Conv2d(1, self.conv1_out_channel, self.conv1_kernel_size, stride=1,
                               padding=(2, 2))

        self.conv2 = nn.Conv2d(self.conv1_out_channel, self.conv2_out_channel, self.conv2_kernel_size, stride=1,
                               padding=(2, 2))

        self.temp = int((((self.image_dim + 2) / 2) + 2) / 2)

        self.fc1 = nn.Linear(self.temp * self.temp * self.conv2_out_channel, self.linear1_out_features)
        self.fc2 = nn.Linear(self.linear1_out_features, self.linear2_out_features)
        self.fc3 = nn.Linear(self.linear2_out_features, self.num_of_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, self.temp * self.temp * self.conv2_out_channel)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

class My_CNNMalware_Model1(nn.Module):
    def __init__(self, image_dim=256, num_of_classes=7):
        super().__init__()

        self.image_dim = image_dim
        self.num_of_classes = num_of_classes

        self.conv1_out_channel = 12
        self.conv1_kernel_size = 3

        self.conv2_out_channel = 16
        self.conv2_kernel_size = 3

        self.linear1_out_features = 120
        self.linear2_out_features = 90
        # self.linear3_out_features = 50

        self.conv1 = nn.Conv2d(1, self.conv1_out_channel, self.conv1_kernel_size, stride=1,
                               padding=(2, 2))

        self.conv2 = nn.Conv2d(self.conv1_out_channel, self.conv2_out_channel, self.conv2_kernel_size, stride=1,
                               padding=(2, 2))

        self.temp = int((((self.image_dim + 2) / 2) + 2) / 2)

        self.fc1 = nn.Linear(self.temp * self.temp * self.conv2_out_channel, self.linear1_out_features)
        self.fc2 = nn.Linear(self.linear1_out_features, self.linear2_out_features)
        # self.fcnew = nn.Linear(self.linear2_out_features, self.linear3_out_features)
        self.fc3 = nn.Linear(self.linear2_out_features, self.num_of_classes)

    def forward(self, X):
        X = F.leaky_relu(self.conv1(X),0.01)
        X = F.max_pool2d(X, 2, 2)
        X = F.leaky_relu(self.conv2(X),0.01)
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, self.temp * self.temp * self.conv2_out_channel)
        X = F.leaky_relu(self.fc1(X),0.01)
        X = F.leaky_relu(self.fc2(X),0.01)
        # X = F.relu(self.fcnew(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

class CNN(Module):
    cmd = 'cnn'
    description = 'This module runs the CNN model'

    def __init__(self):
        super(CNN,self).__init__()
        self.parser = argparse.ArgumentParser(description='CNN Models')
        self.parser.add_argument('--models_structure',action='store_true', help='Show models structure',default=False)
        self.parser.add_argument('--CNNMalware_Model1',action='store_true',help='use  CNNMalware_Model1',default=False)
        self.parser.add_argument('--My_CNNMalware_Model1',action='store_true',help='use My_CNNMalware_Model1',default=False)

    def run(self):
        super(CNN,self).run()
        if self.args.models_structure:
            model = CNNMalware_Model1()
            print(model)
            self.log('info',"模型CNNMalware_Model1的结构如下:")
            self.log('info',str(model))
            model = My_CNNMalware_Model1()
            self.log('info',"模型My_CNNMalware_Model1的结构如下:")
            self.log('info',str(model))
        elif self.args.CNNMalware_Model1:
            if not __sessions__.is_set():
                self.log('info', 'No Sessions')
                return
            current_dir = os.path.dirname(os.path.abspath(__file__))#获取绝对路径
            model_path =current_dir+ "/cnn_canshu/CNNMalware_Model1.pt"#选择参数模型的地址
            dataset_path = __sessions__.current.file.path
            model = CNNMalware_Model1()
            model.load_state_dict(torch.load(model_path))
            transform = torchvision.transforms.Compose([
                    torchvision.transforms.Grayscale(num_output_channels=1),
                    torchvision.transforms.Resize((256,256)),
                    torchvision.transforms.ToTensor()
                ])
            file = open(dataset_path,'rb')
            imageNumber = np.fromfile(file,dtype=np.ubyte)
            filesize = imageNumber.size
            width = 256#设置图片宽度为256
            rem = filesize%width
            # print(rem)
            if rem != 0:
                imageNumber = imageNumber[:-rem]
            height = int(imageNumber.shape[0]/width)
            grayimage = imageNumber.reshape(height,width)
            img = Image.fromarray(grayimage)
            data = transform(img)
            model.eval()
            data = Variable(torch.unsqueeze(data, dim=0).float(), requires_grad=False)
            pred= model(data)
            predicted = torch.max(pred.data, 1)[1]
            anc = str(predicted[0].item())
            dict = {'0':'Backdoor.Win33.MiniDuke.h', '1':'HEURTrojan.Win32.CozyDuke.gen', '2':'HEURTrojan.Win32.Generic','3':'HEURTrojan.Win32.Sofacy.gen','4':'Trojan.Win32.CloudLook.a','5':'Trojan.Win32.Havex.p','6':'Virus.Win32.Pioneer.dx'}
            
            self.log('info', "检测的结果为:"+dict[anc])
            db = Database()
            db.add_tags(__sessions__.current.file.sha256, dict[anc])
            self.log('info',"已经为该结果打上标签!")
        elif self.args.My_CNNMalware_Model1:
            pass
