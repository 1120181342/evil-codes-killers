from viper.common.abstracts import Module
from viper.core.session import __sessions__
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from PIL import Image 
from viper.core.database import Database

class CNNMalware_Model1(nn.Module):
    def __init__(self, image_dim=256, num_of_classes=4):
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

class CNN(Module):
    cmd = 'cnn'
    description = 'This module runs the CNN model'

    def run(self):
        if not __sessions__.is_set():
            self.log('info', 'No Sessions')
            return
        model_path = "/home/cyz/test/CNNMalware_Model1.pt"
        dataset_path = __sessions__.current.file.path
        model = CNNMalware_Model1()
        model.load_state_dict(torch.load(model_path))
        # print(model)
        transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.Resize((256,256)),
                torchvision.transforms.ToTensor()
            ])
        dataset = Image.open(dataset_path)
        data = transform(dataset)
        model.eval()
        # tmp = dataset.imgs
        # print(dataset[0][0])
        # data = dataset[0][0]
        # print(type(img2))
        # print(img2.shape)
        # print(data.shape)
        data = Variable(torch.unsqueeze(data, dim=0).float(), requires_grad=False)
        # print(data.shape)
        pred= model(data)
        # print("\n")
        predicted = torch.max(pred.data, 1)[1]
        anc = str(predicted[0].item())
        dict = {'0':'APT_28', '1':'APT_29', '2':'Dark_Hotel'}
        self.log('info', dict[anc])
        self.log('info',__sessions__.current.file.sha256)
        db = Database()
        db.add_tags(__sessions__.current.file.sha256, dict[anc])
