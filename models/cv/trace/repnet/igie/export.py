# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import argparse
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ArcFC(nn.Module):
    """
    Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output_layer sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self,
                 in_features,
                 out_features,
                 s=30.0,
                 m=0.50,
                 easy_margin=False):
        """
        ArcMargin
        :param in_features:
        :param out_features:
        :param s:
        :param m:
        :param easy_margin:
        """
        super(ArcFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input, p=2), F.normalize(self.weight, p=2))

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))


        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

class RepNet(torch.nn.Module):
    def __init__(self,
                 out_ids,
                 out_attribs):
        """
        Network definition
        :param out_ids:
        :param out_attribs:
        """
        super(RepNet, self).__init__()

        self.out_ids, self.out_attribs = out_ids, out_attribs

        self.conv1_1 = torch.nn.Conv2d(in_channels=3,
                                       out_channels=64,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))
        self.conv1_2 = torch.nn.ReLU(inplace=True)
        self.conv1_3 = torch.nn.Conv2d(in_channels=64,
                                       out_channels=64,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))
        self.conv1_4 = torch.nn.ReLU(inplace=True)
        self.conv1_5 = torch.nn.MaxPool2d(kernel_size=2,
                                          stride=2,
                                          padding=0,
                                          dilation=1,
                                          ceil_mode=False)

        self.conv1 = torch.nn.Sequential(
            self.conv1_1,
            self.conv1_2,
            self.conv1_3,
            self.conv1_4,
            self.conv1_5
        )

        self.conv2_1 = torch.nn.Conv2d(in_channels=64,
                                       out_channels=128,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1)) 
        self.conv2_2 = torch.nn.ReLU(inplace=True)
        self.conv2_3 = torch.nn.Conv2d(in_channels=128,
                                       out_channels=128,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))
        self.conv2_4 = torch.nn.ReLU(inplace=True)
        self.conv2_5 = torch.nn.MaxPool2d(kernel_size=2,
                                          stride=2,
                                          padding=0,
                                          dilation=1,
                                          ceil_mode=False)

        self.conv2 = torch.nn.Sequential(
            self.conv2_1,
            self.conv2_2,
            self.conv2_3,
            self.conv2_4,
            self.conv2_5
        )

        self.conv3_1 = torch.nn.Conv2d(in_channels=128,
                                       out_channels=256,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))
        self.conv3_2 = torch.nn.ReLU(inplace=True)
        self.conv3_3 = torch.nn.Conv2d(in_channels=256,
                                       out_channels=256,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))
        self.conv3_4 = torch.nn.ReLU(inplace=True)
        self.conv3_5 = torch.nn.Conv2d(in_channels=256,
                                       out_channels=256,
                                       kernel_size=(3, 3),
                                       stride=(1, 1),
                                       padding=(1, 1))
        self.conv3_6 = torch.nn.ReLU(inplace=True)
        self.conv3_7 = torch.nn.MaxPool2d(kernel_size=2,
                                          stride=2,
                                          padding=0,
                                          dilation=1,
                                          ceil_mode=False)

        self.conv3 = torch.nn.Sequential(
            self.conv3_1,
            self.conv3_2,
            self.conv3_3,
            self.conv3_4,
            self.conv3_5,
            self.conv3_6,
            self.conv3_7
        )

        self.conv4_1_1 = torch.nn.Conv2d(in_channels=256,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))
        self.conv4_1_2 = torch.nn.ReLU(inplace=True)
        self.conv4_1_3 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))
        self.conv4_1_4 = torch.nn.ReLU(inplace=True)
        self.conv4_1_5 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))
        self.conv4_1_6 = torch.nn.ReLU(inplace=True)
        self.conv4_1_7 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            dilation=1,
                                            ceil_mode=False)

        self.conv4_1 = torch.nn.Sequential(
            self.conv4_1_1,
            self.conv4_1_2,
            self.conv4_1_3,
            self.conv4_1_4,
            self.conv4_1_5,
            self.conv4_1_6,
            self.conv4_1_7
        )

        self.conv4_2_1 = torch.nn.Conv2d(in_channels=256,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))
        self.conv4_2_2 = torch.nn.ReLU(inplace=True)
        self.conv4_2_3 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))
        self.conv4_2_4 = torch.nn.ReLU(inplace=True)
        self.conv4_2_5 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))
        self.conv4_2_6 = torch.nn.ReLU(inplace=True)
        self.conv4_2_7 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            dilation=1,
                                            ceil_mode=False)

        self.conv4_2 = torch.nn.Sequential(
            self.conv4_2_1,
            self.conv4_2_2,
            self.conv4_2_3,
            self.conv4_2_4,
            self.conv4_2_5,
            self.conv4_2_6,
            self.conv4_2_7
        )

        self.conv5_1_1 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))
        self.conv5_1_2 = torch.nn.ReLU(inplace=True)
        self.conv5_1_3 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))
        self.conv5_1_4 = torch.nn.ReLU(inplace=True)
        self.conv5_1_5 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))
        self.conv5_1_6 = torch.nn.ReLU(inplace=True)
        self.conv5_1_7 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            dilation=1,
                                            ceil_mode=False)

        self.conv5_1 = torch.nn.Sequential(
            self.conv5_1_1,
            self.conv5_1_2,
            self.conv5_1_3,
            self.conv5_1_4,
            self.conv5_1_5,
            self.conv5_1_6,
            self.conv5_1_7
        )

        self.conv5_2_1 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))
        self.conv5_2_2 = torch.nn.ReLU(inplace=True)
        self.conv5_2_3 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))
        self.conv5_2_4 = torch.nn.ReLU(inplace=True)
        self.conv5_2_5 = torch.nn.Conv2d(in_channels=512,
                                         out_channels=512,
                                         kernel_size=(3, 3),
                                         stride=(1, 1),
                                         padding=(1, 1))
        self.conv5_2_6 = torch.nn.ReLU(inplace=True)
        self.conv5_2_7 = torch.nn.MaxPool2d(kernel_size=2,
                                            stride=2,
                                            padding=0,
                                            dilation=1,
                                            ceil_mode=False)

        self.conv5_2 = torch.nn.Sequential(
            self.conv5_2_1,
            self.conv5_2_2,
            self.conv5_2_3,
            self.conv5_2_4,
            self.conv5_2_5,
            self.conv5_2_6,
            self.conv5_2_7
        )

        self.FC6_1_1 = torch.nn.Linear(in_features=25088,
                                       out_features=4096,
                                       bias=True)
        self.FC6_1_2 = torch.nn.ReLU(inplace=True)
        self.FC6_1_3 = torch.nn.Dropout(p=0.5)
        self.FC6_1_4 = torch.nn.Linear(in_features=4096,
                                       out_features=4096,
                                       bias=True)
        self.FC6_1_5 = torch.nn.ReLU(inplace=True)
        self.FC6_1_6 = torch.nn.Dropout(p=0.5)

        self.FC6_1 = torch.nn.Sequential(
            self.FC6_1_1,
            self.FC6_1_2,
            self.FC6_1_3,
            self.FC6_1_4,
            self.FC6_1_5,
            self.FC6_1_6
        )

        self.FC6_2_1 = copy.deepcopy(self.FC6_1_1)
        self.FC6_2_2 = copy.deepcopy(self.FC6_1_2)
        self.FC6_2_3 = copy.deepcopy(self.FC6_1_3)
        self.FC6_2_4 = copy.deepcopy(self.FC6_1_4)
        self.FC6_2_5 = copy.deepcopy(self.FC6_1_5)
        self.FC6_2_6 = copy.deepcopy(self.FC6_1_6)

        self.FC6_2 = torch.nn.Sequential(
            self.FC6_2_1,
            self.FC6_2_2,
            self.FC6_2_3,
            self.FC6_2_4,
            self.FC6_2_5,
            self.FC6_2_6
        )

        self.FC7_1 = torch.nn.Linear(in_features=4096,
                                     out_features=1000,
                                     bias=True)

        self.FC7_2 = torch.nn.Linear(in_features=4096,
                                     out_features=1000,
                                     bias=True)

        self.FC_8 = torch.nn.Linear(in_features=2000,
                                    out_features=1024)

        self.attrib_classifier = torch.nn.Linear(in_features=1000,
                                                 out_features=out_attribs)

        self.arc_fc_br2 = ArcFC(in_features=1000,
                                out_features=out_ids,
                                s=30.0,
                                m=0.5,
                                easy_margin=False)
        self.arc_fc_br3 = ArcFC(in_features=1024,
                                out_features=out_ids,
                                s=30.0,
                                m=0.5,
                                easy_margin=False)

        self.shared_layers = torch.nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3
        )

        self.branch_1_feats = torch.nn.Sequential(
            self.shared_layers,
            self.conv4_1,
            self.conv5_1,
        )

        self.branch_1_fc = torch.nn.Sequential(
            self.FC6_1,
            self.FC7_1
        )

        self.branch_1 = torch.nn.Sequential(
            self.branch_1_feats,
            self.branch_1_fc
        )

        self.branch_2_feats = torch.nn.Sequential(
            self.shared_layers,
            self.conv4_2,
            self.conv5_2
        )

        self.branch_2_fc = torch.nn.Sequential(
            self.FC6_2,
            self.FC7_2
        )

        self.branch_2 = torch.nn.Sequential(
            self.branch_2_feats,
            self.branch_2_fc
        )

    def forward(self,
                X,
                branch,
                label=None):
        """
        :param X:
        :param branch:
        :param label:
        :return:
        """
        N = X.size(0)

        branch_1 = self.branch_1_feats(X)
        branch_2 = self.branch_2_feats(X)

        branch_1 = branch_1.view(N, -1)
        branch_2 = branch_2.view(N, -1)
        branch_1 = self.branch_1_fc(branch_1)
        branch_2 = self.branch_2_fc(branch_2)

        assert branch_1.size() == (N, 1000) and branch_2.size() == (N, 1000)

        fusion_feats = torch.cat((branch_1, branch_2), dim=1)

        assert fusion_feats.size() == (N, 2000)

        X = self.FC_8(fusion_feats)

        assert X.size() == (N, 1024)

        return X

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weight", 
                    type=str, 
                    required=True, 
                    help="pytorch model weight.")
    
    parser.add_argument("--output", 
                    type=str, 
                    required=True, 
                    help="export onnx model path.")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    model = RepNet(out_ids=10086, out_attribs=257)

    checkpoint = torch.load(args.weight)
    model.load_state_dict(checkpoint)
    model.eval()

    input_names = ['input']
    output_names = ['output']
    dynamic_axes = {'input': {0: '-1'}, 'output': {0: '-1'}}
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model, 
        dummy_input, 
        args.output, 
        input_names = input_names, 
        dynamic_axes = dynamic_axes, 
        output_names = output_names,
        opset_version=13
    )    
    
    print("Export onnx model successfully! ")