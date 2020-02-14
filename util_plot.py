import numpy as np
import pandas as pd
import time
from util import OpalDataSet
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn

def numpy2prob(data):
    x = torch.tensor(data)
    return F.softmax(x, dim=0)

def boxplot():
    data = OpalDataSet()
    old_test, real_test, fake_test = data.test
    old_test = np.reshape(old_test.values, [-1, 20, old_test.shape[1]])
    old_test = old_test[:, 0, :]  # only use the first record
    real_test = real_test.values
    fake_test = fake_test.values

    old_raw = pd.read_csv('result/old_learned_embedding.csv', header=None).values
    real_raw = pd.read_csv('result/real_learned_embedding.csv', header=None).values
    fake_raw = pd.read_csv('result/fake_learned_embedding.csv', header=None).values

    chunk_size = 300
    num_chunk = real_test.shape[0] // chunk_size

    loss_real_all = []
    loss_fake_all = []
    loss_real_raw_all = []
    loss_fake_raw_all = []
    for chunk in range(num_chunk):
        old_prob = numpy2prob(old_test[chunk * chunk_size: (chunk + 1) * chunk_size])
        real_prob = numpy2prob(real_test[chunk * chunk_size: (chunk + 1) * chunk_size])
        fake_prob = numpy2prob(fake_test[chunk * chunk_size: (chunk + 1) * chunk_size])

        temp1 = old_prob.numpy()
        temp2 = real_prob.numpy()
        temp3 = fake_prob.numpy()

        old_raw_prob = numpy2prob(old_raw[chunk * chunk_size: (chunk + 1) * chunk_size])
        real_raw_prob = numpy2prob(real_raw[chunk * chunk_size: (chunk + 1) * chunk_size])
        fake_raw_prob = numpy2prob(fake_raw[chunk * chunk_size: (chunk + 1) * chunk_size])

        KLdiv = nn.KLDivLoss()
        loss_real = KLdiv(torch.log(real_prob), old_prob).numpy()
        loss_fake = KLdiv(torch.log(fake_prob), old_prob).numpy()

        loss_real_raw = KLdiv(torch.log(real_raw_prob), old_raw_prob).numpy()
        loss_fake_raw = KLdiv(torch.log(fake_raw_prob), old_raw_prob).numpy()

        print(loss_real, loss_fake, loss_real_raw, loss_fake_raw)

        print("learned:", loss_fake / loss_real, "raw:", loss_fake_raw / loss_real_raw)
        if loss_real >= 1e5 or loss_fake >= 1e5:
            continue
        else:
            loss_real_all.append(loss_real)
            loss_fake_all.append(loss_fake)
            loss_real_raw_all.append(loss_real_raw)
            loss_fake_raw_all.append(loss_fake_raw)

    pd.DataFrame(loss_real_all).to_csv('result/ablation/real.csv', header=False, index=False)
    pd.DataFrame(loss_fake_all).to_csv('result/ablation/fake.csv', header=False, index=False)
    pd.DataFrame(loss_real_raw_all).to_csv('result/ablation/real_raw.csv', header=False, index=False)
    pd.DataFrame(loss_fake_raw_all).to_csv('result/ablation/fake_raw.csv', header=False, index=False)

def tsne():
    data = OpalDataSet()
    old_test, real_test, fake_test = data.test
    old_test = np.reshape(old_test.values, [-1, 20, old_test.shape[1]])
    old_test = old_test[:300, 0, :]  # only use the first record
    real_test = real_test.values[:300]
    fake_test = fake_test.values[:300]
    #
    # old_test = pd.read_csv('result/old_learned_embedding.csv', header=None).values[:300]
    # real_test = pd.read_csv('result/real_learned_embedding.csv', header=None).values[:300]
    # fake_test = pd.read_csv('result/fake_learned_embedding.csv', header=None).values[:300]
    # real = np.sum((old_test - real_test)**2, axis=1)
    # fake = np.sum((old_test - fake_test)**2, axis=1)
    # plt.scatter(real, fake)
    # plt.show()
    # pd.DataFrame(real).to_csv('real_kl.csv', header=False, index=False)
    # pd.DataFrame(fake).to_csv('fake_kl.csv', header=False, index=False)
    data = np.concatenate([old_test, real_test, fake_test], axis=0)
    m = data.shape[0]//3
    labels = np.concatenate([np.zeros(m), np.ones(m), 2*np.ones(m)])
    tsne = TSNE(n_components=2, perplexity=45)
    ts = time.time()
    print("start")
    X_2d_raw = tsne.fit_transform(data, labels)
    print("time:", time.time() - ts)
    plt.scatter(X_2d_raw[:m, 0] - X_2d_raw[m:2*m, 0], X_2d_raw[:m,1] - X_2d_raw[m:2*m,1], marker= '^', s = 13, label = 'normal')
    plt.scatter(X_2d_raw[:m, 0] - X_2d_raw[2 * m:2*m+1, 0], X_2d_raw[:m,1] -X_2d_raw[2 * m:2*m+1, 1], s = 13, label='fraud')
    # plt.scatter(X_2d_raw[:m, 0], X_2d_raw[:m,1], s = 10, label = 'historical')
    # plt.scatter(X_2d_raw[m:2*m, 0], X_2d_raw[m:2*m,1], s = 10, label = 'normal')
    # plt.scatter(X_2d_raw[2*m:, 0], X_2d_raw[2*m:,1], s = 10, label = 'fraud')

    plt.xlabel("component 1", fontsize = 14)
    plt.ylabel("component 2", fontsize = 14)
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)
    plt.legend(loc = 'upper left', fontsize=14)
    plt.grid(linestyle='dashed')
    # plt.savefig("result/ablation/tsne_learned_embedding.eps", format="eps")
    plt.savefig("result/ablation/tsne_raw_features.eps", format="eps")
    plt.show()


if __name__ == '__main__':
    tsne()