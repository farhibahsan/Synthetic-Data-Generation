import pickle
import torch.nn.functional as F
from torch.nn.modules.module import register_module_backward_hook
from torch_geometric.nn import ChebConv
from torch.nn import ReLU
import numpy as np
import torch
import argparse
import sys, random

with open("./syntheticdata", "rb") as f:
    data_list = pickle.load(f)



class Net(torch.nn.Module):
    def __init__(self, k):
        super(Net, self).__init__()
        self.conv = ChebConv(2, 2, k)
        self.W = torch.nn.Parameter(torch.rand(3), requires_grad=True)
        self.b = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.relu = ReLU()
        
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        # print(x)
        x_turn = x[:, 2]
        x_tc = x[:, :2]

        x_hidden = self.conv(x_tc, edge_index, edge_weight)
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        x_hidden = self.relu(x_hidden)
        x_lin = torch.column_stack((x_turn, x_hidden))
        u = torch.mv(x_lin, self.W) + self.b

        return F.softmax(u, dim=0)

class CNet(torch.nn.Module):
    def __init__(self, k):
        super(CNet, self).__init__()
        self.W = torch.nn.Parameter(torch.rand(4), requires_grad=True)
        self.b = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        num_alt = int(np.sqrt(len(edge_weight)))
        cf = torch.pow(torch.sum(torch.reshape(edge_weight, (num_alt, num_alt)), 1), self.gamma)
        
        x_lin = torch.column_stack((x, cf))
        u = torch.mv(x_lin, self.W) + self.b

        return F.softmax(u, dim=0)

def train(k):
    model = CNet(k)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()

    random.shuffle(data_list)

    data2_list = [data for data in data_list if len(data.y)==2]
    data3_list = [data for data in data_list if len(data.y)==3]
    data4_list = [data for data in data_list if len(data.y)==4]

    data23_list = data2_list + data3_list
    data24_list = data2_list + data4_list
    data34_list = data3_list + data4_list

    test_sample_num = 50
    data33_list = data3_list[:-test_sample_num]
    data333_list = data3_list[-test_sample_num:]
    data22_list = data2_list[:-test_sample_num]
    data222_list = data2_list[-test_sample_num:]
    data44_list = data4_list[:-test_sample_num]
    data444_list = data4_list[-test_sample_num:]
    data11_list = data_list[:-test_sample_num]
    data111_list = data_list[-test_sample_num:]
    
    training_set = data23_list
    testing_set =  data4_list

    for epoch in range(1000):
        
        for data in training_set:
            optimizer.zero_grad()
            out = model(data)
            # print(out)
            loss = -torch.sum(data.y * torch.log(out + 1e-20))
            # print(loss)
            loss.backward()
            optimizer.step()

    model.eval()
    return {
        "training": sum([int(model(data).argmax().eq(data.y.argmax())) for data in training_set])/len(training_set),
        "testing": sum([int(model(data).argmax().eq(data.y.argmax())) for data in testing_set])/len(testing_set)
    }


def parse_args(args):
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        'k', type = int,
        help='ChevCon hyperparam'
    )

    return parser.parse_known_args(args)[0]

if __name__ == "__main__":
    flags = parse_args(sys.argv[1:])
    result = {
        "training": [],
        "testing": []
    }
    for i in range(10):
        temp = train(flags.k)
        result["training"].append(temp["training"])
        result["testing"].append(temp["testing"])
    
    print("training:")
    for i in range(10):
        print(result["training"][i])
    
    print("testing:")
    for i in range(10):
        print(result["testing"][i])