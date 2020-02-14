import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from util import OpalDataSet
from model import CategoricalEmbedding, Route2Stop, FCN, Encoder, Similarity
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import warnings


warnings.filterwarnings("ignore")

class Model:
    def __init__(self,seq_len = 20, learning_rate = 3e-4):
        device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.seq_len  = seq_len
        time_stamp = time.strftime("%m-%d-%Y_%H:%M:%S", time.localtime())
        print("run on device", device, ",current time:", time_stamp)
        self.writer = SummaryWriter('runs/emb_graph' + time_stamp)

        # define layers
        self.categ_embedding = CategoricalEmbedding().to(device)
        self.r2s_embedding = Route2Stop(vertex_feature=105,edge_feature=112).to(device)
        self.encoder = Encoder(input_size=100, seq_len=seq_len).to(device)
        self.fcn = FCN(input_size=100).to(device)
        self.similarity = Similarity(input_size=30, device=device).to(device)

        # define training parameters
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam([{'params': self.categ_embedding.parameters()},
                                    {'params': self.r2s_embedding.parameters()},
                                    {'params': self.encoder.parameters()},
                                    {'params': self.fcn.parameters()},
                                    {'params': self.similarity.parameters()}],
                                   lr=learning_rate)

    def forward(self, old, real, fake, numer_list, categ_list):

        old = self.categ_embedding(old, numer_list, categ_list, self.device)
        real = self.categ_embedding(real, numer_list, categ_list, self.device)
        fake = self.categ_embedding(fake, numer_list, categ_list, self.device)

        old = self.r2s_embedding(old)
        real = self.r2s_embedding(real)
        fake = self.r2s_embedding(fake)

        old = self.encoder(old)
        real = self.fcn(real)
        fake = self.fcn(fake)

        score_real = self.similarity(old, real)
        score_fake = self.similarity(old, fake)
        return score_real, score_fake

    def metrics(self, score_real, score_fake, label_real_test, label_fake_test):
        y_true = np.concatenate([label_real_test.cpu().numpy(), label_fake_test.cpu().numpy()],axis=0)
        y_pred = torch.cat([torch.argmax(score_real,dim=1, keepdim=True), torch.argmax(score_fake,dim=1, keepdim=True)],dim=0).cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return acc, precision, recall, f1

    def train_and_test(self, data, batch_size = 64, num_epoch = 50):
        #initialize labels before training
        label_real = torch.cat([torch.zeros([batch_size,1]), torch.ones([batch_size, 1])], dim=1).to(self.device)
        label_fake = torch.cat([torch.ones([batch_size, 1]), torch.zeros([batch_size,1])], dim=1).to(self.device)

        old_test, real_test, fake_test = data.test
        test_size = real_test.shape[0]
        label_real_test = torch.ones([test_size, 1]).type(torch.long).to(self.device)
        label_fake_test = torch.zeros([test_size, 1]).type(torch.long).to(self.device)

        for epoch in range(num_epoch):
            total_loss = [0] * len(data)
            total_loss_real = [0] *len(data)
            # training first
            for i, chunk in enumerate(data.train):
                old_chunk, real_chunk, fake_chunk = chunk
                num_batch = real_chunk.shape[0] // batch_size
                for batch in range(num_batch):
                    # get a batch of data pair: (old, real, fake)
                    old_batch = old_chunk.iloc[batch * self.seq_len * batch_size: (batch + 1) * self.seq_len * batch_size, :]
                    real_batch = real_chunk.iloc[batch * batch_size: (batch + 1) * batch_size, :]
                    fake_batch = fake_chunk.iloc[batch * batch_size: (batch + 1) * batch_size, :]

                    score_real, score_fake = self.forward(old_batch,  real_batch, fake_batch, data.numer_list, data.categ_list)

                    loss_real = self.criterion(score_real, label_real)
                    loss_fake = self.criterion(score_fake, label_fake)
                    loss = loss_real + loss_fake

                    total_loss[i] += loss.data
                    total_loss_real[i] += loss_real.data
                    self.optimizer.zero_grad()

                    loss.backward()
                    self.optimizer.step()

                    if (batch + 1) % 100 == 0:
                        print("epoch: %d, chunk: %d, batch: %d, loss: %.3f, real: %.3f, fake: %.3f"
                              % (epoch, i, batch + 1, loss.data, loss_real.data, loss_fake.data))
                total_loss[i] = (total_loss[i] / batch).cpu().numpy()
                total_loss_real[i] = (total_loss_real[i] / batch).cpu().numpy()

            # testing
            score_real, score_fake = self.forward(old_test, real_test, fake_test, data.numer_list, data.categ_list)
            acc, precision, recall, f1 = self.metrics(score_real, score_fake, label_real_test, label_fake_test)
            print("test acc: %.4f" % acc)
            self.writer.add_scalar('testing accuracy', acc, epoch)
            self.writer.close()
            # print result and save loss in tensorboard
            print("epoch: %d, average loss: %.4f" % (epoch, np.mean(total_loss)))
            self.writer.add_scalars('training loss', {'overall': np.mean(total_loss),
                                                      'good': np.mean(total_loss_real)}, epoch)
            self.writer.close()
            return acc, precision, recall, f1


if __name__ == '__main__':
    data = OpalDataSet()
    precisions = []
    recalls = []
    f1s = []
    accs = []
    # test for 10 times
    for test_time in range(10):
        print ("test epoch:", test_time+1)
        model = Model()
        acc,precision, recall, f1 = model.train_and_test(data)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accs.append(acc)
    # pd.DataFrame(precisions).to_csv('result/precision.csv', header=False, index=False)
    # pd.DataFrame(recalls).to_csv('result/recall.csv', header=False, index=False)
    # pd.DataFrame(f1s).to_csv('result/f1.csv', header=False, index=False)
    # pd.DataFrame(accs).to_csv('result/acc.csv', header=False, index=False)













