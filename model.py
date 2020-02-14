import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CategoricalEmbedding(nn.Module):
    def __init__(self):
        super(CategoricalEmbedding, self).__init__()
        self.embedding_dict = {
            "TS_TYP_CD":(5,3), "IMTT_CD":(8, 4), "ROUTE_ID": (1218, 50),"ROUTE_VAR_ID": (688, 50),
            "RUN_DIR_CD": (4, 2), "TAG1_TS_PC": (332, 50), "TAG1_TS_NUM": (831, 50), }
        self.embs = nn.ModuleList([nn.Embedding(value[0], value[1]) for value in self.embedding_dict.values()])

    def forward(self,x, numer_list, categ_list, device):
        x_num = torch.Tensor(x[numer_list].values).to(device)
        x = torch.LongTensor(x[categ_list].values).to(device)
        output1 = [emb(x[:,i]) for i, emb in enumerate(self.embs)]
        output2 = [self.embs[-2](x[:,-2]), self.embs[-1](x[:, -1])] # 'TAG2_TS_PC','TAG2_TS_NUM' should share same embedding with TAG1_**
        output = torch.cat(output1 + output2, dim=1)
        output = torch.cat([output, x_num], dim = 1)
        # print ("embedding:",output[0,:])
        return output

class Route2Stop(nn.Module):
    def __init__(self, vertex_feature= 106, edge_feature= 106, output_size=50):
        super(Route2Stop, self).__init__()
        self.edge2vertex = nn.Sequential(nn.Linear(edge_feature, vertex_feature), nn.ReLU())
        self.graph_conv = nn.Sequential(nn.Linear(vertex_feature, output_size), nn.ReLU())

    def forward(self,data):
        # data is a tensor after categ_embedding
        # node1 feature: TAG1_TS_NUM, TAG1_TS_NUM_PC, TAG1_LAT, TAG1_LONG, TAG1_TM_HOUR/MIN/MONTH/DAY
        # edge feature: ROUTE_ID/ ROUTE_VAR_ID / ROUTE_VAR_TYP_CD / RUN_DIR_CD
        node1_feature = list(range(109, 209)) + list(range(-10, -5)) # to split node and edge features before they were embedded
        node2_feature = list(range(209, 309)) + list(range(-5, 0))
        edge_feature = list(range(109)) + list(range(-13,-10))
        node1_data = data[:, node1_feature]
        node2_data = data[:, node2_feature]
        edge_data = data[:, edge_feature]

        e2v = self.edge2vertex(edge_data)
        # print("graph-e2v:",e2v[0,:])
        v_i = self.graph_conv(node1_data * e2v)
        # print("graph-v1:",v_i[0,:])
        v_j = self.graph_conv(node2_data * e2v)
        # print("graph-v2",v_j[0,:])
        graph_data = torch.cat([v_i, v_j], dim=1)
        return graph_data

class Encoder(nn.Module):
    def __init__(self, input_size = 0, seq_len = 20, hidden_size = 50, output_size = 30, sequence = 'LSTM'):
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        if sequence == 'LSTM':
            self.lstm1 = nn.LSTM(input_size,hidden_size)
        else:
            self.lstm1 = nn.GRU(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size,output_size)


    def forward(self, x):
        if len(x.shape) != 3:
            x = x.view(-1, 20, x.size(1))
            x = x[:, :self.seq_len, :]
            x = x.transpose(0,1)
        x,_ = self.lstm1(x)
        # x = x[-1,:,:]
        # print ("lstm-1:",x[0,:])
        x = self.fc1(x)
        # print ("lstm-2:",x[0,:])
        return x

class FCN(nn.Module):
    def __init__(self, input_size = 0, output_size = 30):
        super(FCN, self).__init__()
        hidden_size = [50, 20]
        self.fc2 = nn.Linear(input_size, hidden_size[0])
        self.fc3 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc4 = nn.Linear(hidden_size[1], output_size)

    def forward(self, x):
        x = F.relu(self.fc2(x))
        # print("fc-1:",x[0,:])
        x = F.relu(self.fc3(x))
        # print("fc-2:",x[0,:])
        x = F.relu(self.fc4(x))
        # print("fc-3:",x[0,:])
        return x

class Similarity(nn.Module):
    def __init__(self, input_size = 0, output_size = 2, device = None):
        super(Similarity, self).__init__()
        hidden_size  = 1
        self.fc5 = nn.Linear(input_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)
        self.M1 = Variable(torch.rand((input_size, input_size)), requires_grad=True).to(device)
        self.b1 = Variable(torch.tensor([0.0]), requires_grad=True).to(device)
        self.M2 = Variable(torch.rand((input_size, input_size)), requires_grad=True).to(device)
        self.b2 = Variable(torch.tensor([0.0]), requires_grad=True).to(device)

    def forward(self, old,new, return_old = False):
        old = torch.transpose(old, 0,1)
        attn = self.fc5(old)
        attn = torch.softmax(attn, dim=1)
        old = old * attn
        old = torch.sum(old, dim=1)
        score_pos = torch.matmul(torch.matmul(new, self.M1), old.T) + self.b1
        score_pos = torch.diag(score_pos)
        score_neg = torch.matmul(torch.matmul(new, self.M2), old.T) + self.b2
        score_neg = torch.diag(score_neg)
        score = torch.softmax(torch.stack([score_pos, score_neg]).T, dim=1)
        if return_old:
            return old
        else:
            return score