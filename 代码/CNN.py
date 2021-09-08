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
import json
from socket import *
import time


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

class My_CNNMalware_lenet5(nn.Module):
    def __init__(self, image_dim=256, num_of_classes=7):
        super().__init__()

        self.image_dim = image_dim
        self.num_of_classes = num_of_classes

        self.conv1_out_channel = 6
        self.conv1_kernel_size = 5

        self.conv2_out_channel = 16
        self.conv2_kernel_size = 5

        self.conv3_out_channel=120
        self.conv3_kernel_size=5

        self.linear1_out_features = 84
        # self.linear2_out_features = 90
        # self.linear3_out_features = 50

        self.conv1 = nn.Conv2d(1, self.conv1_out_channel, self.conv1_kernel_size, stride=1,
                               padding=(2, 2))

        self.conv2 = nn.Conv2d(self.conv1_out_channel, self.conv2_out_channel, self.conv2_kernel_size, stride=1,
                               padding=(2, 2))
        self.conv3 = nn.Conv2d(self.conv2_out_channel, self.conv3_out_channel, self.conv3_kernel_size, stride=1,
                               padding=(2, 2))

        # self.temp = 401

        self.fc1 = nn.Linear(120*64*64, self.linear1_out_features)
        self.fc2 = nn.Linear(self.linear1_out_features, self.num_of_classes)
        # self.fcnew = nn.Linear(self.linear2_out_features, self.linear3_out_features)
        # self.fc3 = nn.Linear(self.linear2_out_features, self.num_of_classes)

    def forward(self, X):
        X = self.conv1(X)
        X = F.sigmoid(F.max_pool2d(X, 2, 2))
        X = self.conv2(X)
        X = F.sigmoid(F.max_pool2d(X, 2, 2))
        X = self.conv3(X)
        X = X.view(-1, 120*64*64)
        X = self.fc1(X)
        X = self.fc2(X)
        # X = F.relu(self.fcnew(X))
        # X = self.fc3(X)
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
        self.parser.add_argument('--My_CNNMalware_lenet5',action='store_true',help='use My_CNNMalware_lenet5',default=False)
        self.parser.add_argument('--Custom_predict',action='store_true',help='Use custom detection scheme',default=False)
        self.parser.add_argument('--upload',action='store_true',help='upload this malware',default=False)
        self.malware_dict =  {'0':'Backdoor.Win33.MiniDuke.h', '1':'HEURTrojan.Win32.CozyDuke.gen', '2':'HEURTrojan.Win32.Generic','3':'HEURTrojan.Win32.Sofacy.gen','4':'Trojan.Win32.CloudLook.a','5':'Trojan.Win32.Havex.p','6':'Virus.Win32.Pioneer.dx'}

    def predict_malware(self,model,model_path):
            model.load_state_dict(torch.load(model_path))
            transform = torchvision.transforms.Compose([
                    torchvision.transforms.Grayscale(num_output_channels=1),
                    torchvision.transforms.Resize((256,256)),
                    torchvision.transforms.ToTensor()
                ])
            dataset_path = __sessions__.current.file.path
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
            gailv = torch.softmax(pred.data,dim=1).data
            max_gailv = 0.0
            for i in range(len(gailv[0])):
                self.log('info',f'恶意代码为{self.malware_dict[str(i)]}的概率为{str(gailv[0][i].item())}')
                if max_gailv < gailv[0][i].item():
                    max_gailv = gailv[0][i].item()
            predicted = torch.max(pred.data, 1)[1]
            anc = str(predicted[0].item())
            if max_gailv < 0.5:
                result = 'other_malware'
            else:
                result = self.malware_dict[anc]
            return result

    def use_CNNMalware_Model1(self,model_path):
            if not __sessions__.is_set():
                self.log('info', 'No Sessions')
                return
            model = CNNMalware_Model1()
            pred = self.predict_malware(model,model_path)
            return pred


    def use_My_CNNMalware_Model1(self,model_path):
            if not __sessions__.is_set():
                self.log('info', 'No Sessions')
                return
            model = My_CNNMalware_Model1()
            pred = self.predict_malware(model,model_path)
            return pred

    def use_My_CNNMalware_lenet5(self,model_path):
            if not __sessions__.is_set():
                    self.log('info', 'No Sessions')
                    return
            model = My_CNNMalware_lenet5()
            pred = self.predict_malware(model,model_path)
            return pred

    def use_custom_method(self):
        json_path = '/home/wwh/dataset/cnn_params.json'
        with open(json_path,"r") as file:
                datas_json = file.read()
        datas = json.loads(datas_json)
        try:
            predict_ways = []
            for data in datas.values():
                predict_ways.append(data)
            print(predict_ways)
            for predict_way in predict_ways:
                module_name = predict_way['model_name']
                module_path = predict_way['model_params_path']
                self.log('info',f'使用的模型名称为{module_name}')
                self.log('info',f'模型的参数路径在{module_path}')
                if module_name == 'CNNMalware_Model1':
                    pred = self.use_CNNMalware_Model1(module_path)
                elif module_name == 'My_CNNMalware_Model1':
                    pred = self.use_My_CNNMalware_Model1(module_path)
                elif module_name == 'My_CNNMalware_lenet5':
                    pred = self.use_My_CNNMalware_lenet5(module_path)
                else:
                    self.log('info',f'模型{module_name}不是内置模型，无法进行检测!')
                self.log('info',f'该模型的检测结果为{pred}')
                self.log('info','--------------------------------------------------------\n')
        except:
            self.log('info',"cnn_params.json存在错误或者模型参数不适合该模型!")               

    def print_result(self,pred):
        if pred == "other_malware":
            self.log('info',"由于检测结果均未达到指定标准，固不属于以上7类恶意代码家族!")
            self.log('info',"该恶意代码属于其它类别!")
            db = Database()
            db.add_tags(__sessions__.current.file.sha256,"Other_malwares")
        else:
            self.log('info', "检测的结果为:"+pred)
            db = Database()
            db.add_tags(__sessions__.current.file.sha256,pred)
            self.log('info',"已经为该结果打上标签!")

    def upload(self):
        db = Database()
        tags = db.list_tags_for_malware(__sessions__.current.file.sha256)
        self.log('info','正在检测结果!')
        tags_anc = []
        for i in range(len(tags)):
            tags_anc.append(tags[i].to_dict()['tag'])
        if len(tags_anc) > 1:
            self.log('info','该恶意代码存在多个标签，请保留其中一个!')
        elif len(tags_anc) == 0:
            self.log('info','该恶意代码还没有标签，请进行检测后再上传!')
        else:
            anc = tags_anc[0]
            malware_name = __sessions__.current.file.sha256
            self.to_opengauss(malware_name,anc)
            self.log('info',f'该恶意代码sha256值为{malware_name}已经成功上传,使用的标签为{anc}!')
            

    def to_opengauss(self,malware_name,anc):
        client = socket(AF_INET, SOCK_STREAM)
        ip_port = ('119.3.254.185', 8080)
        buffSize = 1024
        client.connect(ip_port)
        self.log('info','connecting......')

        select = "10"
        client.send(bytes(select, "utf-8"))
        time.sleep(5)
        msg = malware_name + '@' + anc
        client.send(bytes(msg, "utf-8"))

        completed = client.recv(buffSize).decode("utf-8")
        if completed == "1":
            self.log('info', '上传成功')
        
        select = "-1"
        client.send(bytes(select, "utf-8"))
        time.sleep(3)
        self.log('info', '退出系统！')
        client.close()

    def run(self):
        super(CNN,self).run()
        if self.args.models_structure:
            model = CNNMalware_Model1()
            self.log('info',"模型CNNMalware_Model1的结构如下:")
            self.log('info',str(model))
            model = My_CNNMalware_Model1()
            self.log('info',"模型My_CNNMalware_Model1的结构如下:")
            self.log('info',str(model))
            model = My_CNNMalware_lenet5()
            self.log('info',"模型My_CNNMalware_lenet5的结构如下:")
            self.log('info',str(model))
        elif self.args.CNNMalware_Model1:
            self.log('info','使用的模型是CNNMalware_Model1:')
            current_dir = os.path.dirname(os.path.abspath(__file__))#获取绝对路径
            model_path =current_dir+ "/cnn_canshu/CNNMalware_Model1.pt"#选择参数模型的地址
            pred = self.use_CNNMalware_Model1(model_path)
            self.print_result(pred)
        elif self.args.My_CNNMalware_Model1:
            self.log('info','使用的模型是My_CNNMalware_Model1:')
            current_dir = os.path.dirname(os.path.abspath(__file__))#获取绝对路径
            model_path =current_dir+ "/cnn_canshu/My_CNNMalware_Model1.pt"#选择参数模型的地址
            pred = self.use_My_CNNMalware_Model1(model_path)
            self.print_result(pred)
        elif self.args.My_CNNMalware_lenet5:
            self.log('info','使用的模型是My_CNNMalware_lenet5:')
            current_dir = os.path.dirname(os.path.abspath(__file__))#获取绝对路径
            model_path =current_dir+ "/cnn_canshu/My_CNNMalware_lenet5.pt"#选择参数模型的地址
            pred = self.use_My_CNNMalware_lenet5(model_path)
            self.print_result(pred)
        elif self.args.Custom_predict:
            self.log('info','使用自定义的检测方案')
            self.use_custom_method()
        elif self.args.upload:
            self.upload()

        

        
