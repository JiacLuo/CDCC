from torch import nn
import torch
from torch.nn.functional import normalize
from algorithm.CDCC.lstm import LSTM

class CDCC(nn.Module):
    def __init__(self, configs):
        super(CDCC, self).__init__()
        # Time Encoder
        self.time_encoder=LSTM(configs.input_size, configs.hidden_size, configs.num_layers, configs.output_size,configs.dropout_rate)
        # Frequency Encoder
        self.conv_block1_f = nn.Sequential(
            nn.Conv1d(configs.input_channels, 16, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )
        self.conv_block2_f = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3_f = nn.Sequential(
            nn.Conv1d(32, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        #Instance-level projection layers
        self.instance_projector_t = nn.Sequential(
            nn.Linear(configs.output_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.instance_projector_f = nn.Sequential(
            nn.Linear(configs.CNNoutput_channel, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        #Cluster-level projection layer
        self.cluster_projector_t = nn.Sequential(
            nn.Linear(configs.output_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, configs.class_num),
            nn.Softmax(dim=1)
        )
        self.cluster_projector_f = nn.Sequential(
            nn.Linear(configs.CNNoutput_channel, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, configs.class_num),
            nn.Softmax(dim=1)
        )
        self.Dropout1 = nn.Dropout(p=configs.dropout_rate)
        self.Dropout2 = nn.Dropout(p=configs.dropout_rate)
    def forward(self, x_in, x_in_aug,mode='t'):
        #Temporal Domain
        if mode=='t':
            x = self.time_encoder.forward(x_in)
            x_aug = self.time_encoder.forward(x_in_aug)
            h_x = x.reshape(x.shape[0], -1)
            h_aug = x_aug.reshape(x_aug.shape[0], -1)
            h_x = self.Dropout1(h_x)
            h_aug = self.Dropout2(h_aug)

            """Instance-level mapping and cluster-level mapping"""
            z_i_x = normalize(self.instance_projector_t(h_x), dim=1)
            z_c_x = self.cluster_projector_t(h_x)
            z_i_aug = normalize(self.instance_projector_t(h_aug),dim=1)
            z_c_aug = self.cluster_projector_t(h_aug)
        # Frequency Domain
        if mode == 'f':
            x = self.conv_block1_f(x_in)
            x = self.conv_block2_f(x)
            x = self.conv_block3_f(x)
            x_aug = self.conv_block1_f(x_in_aug)
            x_aug = self.conv_block2_f(x_aug)
            x_aug = self.conv_block3_f(x_aug)
            h_x = x.reshape(x.shape[0], -1)
            h_aug = x_aug.reshape(x_aug.shape[0], -1)

            z_i_x = normalize(self.instance_projector_f(h_x), dim=1)  # z_i(batch*class_num) 对两种数据增强方式的列向量进行实例级对比聚类，并归一化
            z_c_x = self.cluster_projector_f(h_x)

            z_i_aug = normalize(self.instance_projector_f(h_aug),
                                dim=1)  # z_i(batch*class_num) 对两种数据增强方式的列向量进行实例级对比聚类，并归一化
            z_c_aug = self.cluster_projector_f(h_aug)
        return h_x, z_i_x, z_c_x, h_aug, z_i_aug, z_c_aug
    def forward_cluster(self, x_in_t):
        #After passing through the representation network, cluster-level mapping of the time series yields clustering results
        x = self.time_encoder.forward(x_in_t)
        h_time = x.reshape(x.shape[0], -1)
        z_time = self.cluster_projector_t(h_time)
        c=torch.argmax(z_time,dim=1)
        return c


