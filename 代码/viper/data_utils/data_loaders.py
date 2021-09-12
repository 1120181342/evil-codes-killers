import random
from torch.utils.data import DataLoader
import torchvision
import torchtext
from viper.config.config import *
import numpy as np


def get_image_datapath(image_dim, check_exist=True):
    if image_dim not in supported_image_dims:
        raise Exception("Unknown Image dim given")

    dir_name = os.path.join(ORG_DATASET_ROOT_PATH, ORG_DATASET_DIR_NAME + '_width_' + str(image_dim))

    if not check_exist:
        return dir_name

    if os.path.isdir(dir_name):
        return dir_name

    raise Exception("Data dir for Image dim {image_dim} missing".format(image_dim=image_dim))


def get_opcode_datapath(opcode_len, check_exist=True):
    if check_exist:
        if opcode_len not in supported_opcode_lens:
            raise Exception("Unknown opcode len given")

    opcode_len_str = '_' + str(opcode_len)
    if opcode_len == -1:
        opcode_len_str = ''

    train_split_json = os.path.join('org_dataset_opcodes_train' + opcode_len_str + '.json')
    test_split_json = os.path.join('org_dataset_opcodes_test' + opcode_len_str + '.json')
    combined_path_for_json = 'org_dataset_opcodes_split' + opcode_len_str
    exist = False

    if os.path.isfile(os.path.join(ORG_DATASET_ROOT_PATH, train_split_json)) and \
            os.path.isfile(os.path.join(ORG_DATASET_ROOT_PATH, test_split_json)) and \
            os.path.isdir(os.path.join(ORG_DATASET_ROOT_PATH, combined_path_for_json)):
        exist = True

    data_path = dict()
    data_path['train_split_json'] = train_split_json
    data_path['test_split_json'] = test_split_json
    data_path['combined_path_for_json'] = combined_path_for_json

    if not check_exist:
        return data_path
    if check_exist and exist:
        return data_path

    print(f'train_split_json : {train_split_json}')
    print(f'test_split_json : {test_split_json}')
    raise Exception("Data dir for opcode len {opcode_len} missing".format(opcode_len=opcode_len))


def get_image_data_loaders(data_path=None, image_dim=64, train_split=0.8, batch_size=256,
                           convert_to_rgb=False, pretrained_image_dim=64, conv1d_image_dim_w=1024):
    workers_count = min(int(CPU_COUNT * 0.80), batch_size)

    image_dim_h = image_dim
    image_dim_w = image_dim

    if image_dim == 0:
        # Conv1D case
        image_dim_h = 1
        image_dim_w = conv1d_image_dim_w

    transform = None
    if convert_to_rgb:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((pretrained_image_dim, pretrained_image_dim)),
            torchvision.transforms.Lambda(lambda image: image.convert('RGB')),
            torchvision.transforms.ToTensor()
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize((image_dim_h, image_dim_w)),
            torchvision.transforms.ToTensor()
        ])

    dataset = torchvision.datasets.ImageFolder(data_path, transform=transform)
    dataset_len = len(dataset)
    # dataset_len = 1000
    #进行初始化测试集基准，确保每个标签可以被检测一次
    dataset_index = 0
    init_test_index=[]
    for i in range(len(dataset.classes)):
        while(1):
            if dataset.imgs[dataset_index][1] == i:
                init_test_index.append(dataset_index)
                break
            else:
                dataset_index+=1

    indices = list(range(dataset_len))
    random.shuffle(indices)
    split = int(np.floor(train_split * dataset_len))
    
    train_list = indices[:split]
    test_list = indices[split:dataset_len]
#确保测试集包含每一个标签
    for i in init_test_index:
        if i not in test_list:
            test_list.append(i)
    print("------")
    print(init_test_index)
    print(test_list)
    print("------")
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(train_list),
                                               num_workers=0)

    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                 test_list),
                                             num_workers=0)

    train_set_len = len(train_loader) * batch_size
    val_set_len = len(val_loader) * batch_size
    class_names = dataset.classes
    num_of_classes = len(dataset.classes)

    return train_loader, val_loader, dataset_len, class_names


def get_opcode_data_loaders(data_path=None, opcode_len=500, batch_size=512):
    TEXT = torchtext.data.Field()
    LABEL = torchtext.data.LabelField()

    fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}

    train_data, test_data = torchtext.data.TabularDataset.splits(
        path=ORG_DATASET_ROOT_PATH,
        train=data_path['train_split_json'],
        test=data_path['test_split_json'],
        format='json',
        fields=fields
    )
    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)

    train_iterator, test_iterator = torchtext.data.BucketIterator.splits((train_data, test_data),
                                                                         batch_size=batch_size,
                                                                         sort=False,
                                                                         shuffle=True,
                                                                         repeat=False,
                                                                         device='cpu')

    dataset_len = len(train_data) + len(test_data)
    class_names = list(LABEL.vocab.stoi.keys())
    text_vocal_len = len(TEXT.vocab)
    label_vocab_len = len(LABEL.vocab)

    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]

    return train_iterator, test_iterator, dataset_len, class_names, text_vocal_len, label_vocab_len, pad_idx
