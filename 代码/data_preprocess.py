from viper.common.abstracts import Module
import argparse
import sys
from socket import*
import struct
import json
import os
import time
import zipfile
from viper.config.config import *
from viper.data_utils.extract_pe_features import *
from viper.data_utils.bin_to_img import *
from viper.data_utils.extract_opcode import *
from viper.data_utils.misc import *
from viper.data_utils.data_loaders import *



class Data_preprocess(Module):
    cmd = 'data_preprocess'
    description = 'This module runs data_preprocess'

    def __init__(self):
        super(Data_preprocess,self).__init__()
        self.parser = argparse.ArgumentParser(description='Process the Malware data')
        self.parser.add_argument('--load_datas',action='store_true',help='Load the dataset from opengauss',default=False)
        self.parser.add_argument('--extract_pe_features', action='store_true', help='Extract features from PE format',
                        default=False)
        self.parser.add_argument('--bin_to_img', action='store_true', help='Generate image files from malware binaries',
                        default=False)
        self.parser.add_argument('--extract_opcodes', action='store_true', help='Extract opcodes from malware binaries',
                        default=False)
        self.parser.add_argument('--count_samples', action='store_true', help='Count all sample files for all experiments',
                        default=False)
        self.parser.add_argument('--split_opcodes', action='store_true', help='split opcodes into train-test for TorchText',
                        default=False)

        self.parser.add_argument('--latex_format', action='store_true', help='Normalize Conf. matrix and save for latex',
                        default=False)
    
    def load_datas(self):
        client = socket(AF_INET, SOCK_STREAM)
        ip_port = ('119.3.254.185', 8080)
        buffSize = 1024
        client.connect(ip_port)
       #  self.log('info',"connecting...")
        for i in range(0, 7):
            select = str(i)
            client.send(bytes(select, "utf-8"))
            self.log('info', select)
            head_struct = client.recv(4)
            head_len = struct.unpack('i', head_struct)[0]
            data = client.recv(head_len)
            head_dir = json.loads(data.decode('utf-8'))
            filesize_b = head_dir["fileSize"]
            filename = head_dir["fileName"]
            FILEPATH_tmp = FILEPATH +"/"+filename
            self.log('info',"文件路径：" + FILEPATH_tmp)
            # if not os.path.exists(FILEPATH_tmp):
            #     os.makedirs(FILEPATH_tmp)
            # FILEPATH_tmp += '/'
            recv_len = 0
            recv_mesg = b''
            f = open("%s%s" % (FILEPATH, filename), "wb")

            while recv_len < filesize_b:
                if filesize_b - recv_len > buffSize:
                    # 假设未上传的文件数据大于最大传输数据
                    recv_mesg = client.recv(buffSize)
                    f.write(recv_mesg)
                    recv_len += len(recv_mesg)
                else:
                    # 需要传输的文件数据小于最大传输数据大小
                    recv_mesg = client.recv(filesize_b - recv_len)
                    recv_len += len(recv_mesg)
                    f.write(recv_mesg)
                    f.close()
                    # self.log('info',"文件接收完毕！")
            completed = "1"
            client.send(bytes(completed, "utf-8"))
            self.log('info',filename+"下载完毕！")
            tmp = filename[:-4]
            unzip_path = FILEPATH + tmp
            if zipfile.is_zipfile(FILEPATH_tmp):
                FILE = zipfile.ZipFile(FILEPATH_tmp, 'r')
                for file in FILE.namelist():
                    FILE.extract(file, unzip_path)
                FILE.close()
                self.log('info',filename+ "解压完毕！")
            else:
                self.log('info', 'This is not zip')
            
            time.sleep(2)
            if os.path.exists(FILEPATH_tmp):
                os.remove(FILEPATH_tmp)
            else:
                self.log('info', 'The file does not exist')
        select = "-1"
        client.send(bytes(select, "utf-8"))
        time.sleep(3)
        self.log('info', '退出系统！')        
        client.close()
        # client.connect(ip_port)
        # self.log('info',"ok")

    def run(self):
        super(Data_preprocess, self).run()
        max_files = 0  # set 0 to process all files or set a specific number
        if self.args.bin_to_img:
            list_of_widths = [0, 1, 64, 128, 256, 512, 1024]
            for width in list_of_widths:
                self.log('info','正在加载width为'+str(width)+'的图片')
                convert_bin_to_img(ORG_DATASET_PATH, width, max_files=max_files)
            self.log('info','加载结束!\n')
        elif self.args.load_datas:
            self.load_datas()
            self.log('info','数据下载成功!\n')
        elif self.args.extract_pe_features:
            extract_pe_features(ORG_DATASET_PE_FEATURES_CSV, ORG_DATASET_COUNT_PE_FEATURES_CSV, ORG_DATASET_PATH,max_files=max_files)
        elif self.args.extract_opcodes:
            process_opcodes_bulk(ORG_DATASET_PATH, max_files=max_files)
        elif self.args.count_samples:
            count_dataset(ORG_DATASET_PATH, ORG_DATASET_COUNT_CSV)
            count_dataset(ORG_DATASET_OPCODES_PATH, ORG_DATASET_COUNT_OPCODES_PATH)
            count_dataset(get_image_datapath(image_dim=256), ORG_DATASET_COUNT_IMAGES_CSV)
        elif self.args.split_opcodes:
            list_of_opcode_lens = [10, 20, 50, 100, 500, 1000, 2000, 5000]
            for opcode_len in list_of_opcode_lens:
                process_split_opcodes(ORG_DATASET_OPCODES_PATH, opcode_len=opcode_len)
        elif self.args.latex_format:
                    # tuple -> log_date_dir , experiment
            data_list = [("25-May-2020_22_44_37", "experiment_14"),
                     ("13-Jun-2020_16_49_09", "experiment_29"),
                     ("09-Jun-2020_20_42_39", "conv1d_experiment_65"),
                     ("14-Jun-2020_09_03_12", "experiment_18"),
                     ("06-Jun-2020_22_13_17", "rnn_experiment_22"),
                     ("06-Jun-2020_22_13_17", "rnn_experiment_46"),
                     ("13-Jun-2020_21_04_18", "tl_experiment_1"),
                     ("12-Jun-2020_21_54_44", "XGB_experiment_1"),
                     ("12-Jun-2020_21_54_44", "Knn_experiment_1"),
                     ("12-Jun-2020_21_54_44", "RandomForest_experiment_1")
                     ]
            process_cf_for_latex(data_list)
        else:
            self.log('error', 'At least one of the parameter is required')
            self.usage()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Process the Malware data')

#     parser.add_argument('--extract_pe_features', action='store_true', help='Extract features from PE format',
#                         default=False)
#     parser.add_argument('--bin_to_img', action='store_true', help='Generate image files from malware binaries',
#                         default=False)
#     parser.add_argument('--extract_opcodes', action='store_true', help='Extract opcodes from malware binaries',
#                         default=False)
#     parser.add_argument('--count_samples', action='store_true', help='Count all sample files for all experiments',
#                         default=False)
#     parser.add_argument('--split_opcodes', action='store_true', help='split opcodes into train-test for TorchText',
#                         default=False)

#     parser.add_argument('--latex_format', action='store_true', help='Normalize Conf. matrix and save for latex',
#                         default=False)

#     args = parser.parse_args()

#     if len(sys.argv) < 2:
#         parser.print_usage()
#         sys.exit(1)

#     main()

