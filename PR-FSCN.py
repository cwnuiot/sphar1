import os
import math
import dgl
from torch.utils.data import Dataset
import torch
from torch import nn
from SAGPoolh import Classifier
from SAGPoolh import RFDataset
from sklearn.metrics import confusion_matrix

device = torch.device('cpu')
filenames = os.listdir(r'test')
alldata = []
labeldata = []
fenge = []
ztfenge = []
depth = 1
BATCH_SIZE = 32
label_tem=0
def getdata(path):
    filenames = os.listdir(path)
    alllabel = []
    for i in range(len(filenames)):
        zpath = 'test/' + filenames[i]
        with open(zpath, "r") as f:
            lines=f.readlines()
            newdata = []
            for id in range(0, 16):
                newdata.append([])
            for i in range(len(lines)):
                line = lines[i] .strip('\n')  # 去掉列表中每一个元素的换行符
                x = line.split()
                data = []
                if len(x)<7:
                    alllabel.append(label_tem)
                    alldata.append(newdata)
                    newdata = []
                    for id in range(0, 16):
                        newdata.append([])
                if len(x)==7:
                    id=int(x[1])
                    newdata[id].append(float(x[3])/1410)
                    newdata[id].append(float(x[4])/(-81.5))
                    newdata[id].append(float(x[5])/(6.283))
                    #print(len(newdata),len(newdata[0]))
                    label_tem=int(x[6])
    print(len(alllabel),len(alldata))#HLB: 844
    allalldata = []
    for j in alldata:
        for q in range(len(j)):
            if len(j[q]) < (150 // depth):
                x = 150 // depth - len(j[q])
                buchong = [0] * x
                j[q] = j[q] + buchong
            if len(j[q]) > 150 // depth:
                j[q] = j[q][:150 // depth]
        allalldata.append(j)
    return allalldata, alllabel


data, label = getdata('test')
print(len(data))
print(len(data[0]))
print(len(data[0][0]))
# print(label)
# print(data[0])
'''data=np.array(data)
label=np.array(label)'''
import random


def split_train_test(data, label, test_ratio):
    # 设置随机数种子，保证每次生成的结果都是一样的
    """random.seed(42)
    random.shuffle(data)
    random.seed(42)
    random.shuffle(label)
    test_set_size = int(len(data) * test_ratio)
    newdata, newlabel = [], []
    for k in range(21):
        newdata.append([]), newlabel.append([])
    for i in range(len(label)):
        newdata[label[i]].append(data[i])
    train_data, test_data, train_label, test_label = [], [], [], []
    print('ssr', len(newdata[0][0]))
    for i in range(len(newdata)):
        for j in range(len(newdata[i])):
            if j < int(len(newdata[i]) * test_ratio):
                test_data.append(newdata[i][j])
                test_label.append(i)
            else:
                train_data.append(newdata[i][j])
                train_label.append(i)
    random.seed(42)
    random.shuffle(train_data)
    random.seed(42)
    random.shuffle(train_label)
    test_data = torch.Tensor(test_data)
    test_label = torch.Tensor(test_label)
    train_data = torch.Tensor(train_data)
    train_label = torch.Tensor(train_label)"""
    random.seed(42)
    random.shuffle(data)
    random.seed(42)
    random.shuffle(label)
    test_set_size = int(len(data) * test_ratio)
    # print(test_set_size)
    test_data = data[:test_set_size]
    test_label = label[:test_set_size]
    train_data = data[test_set_size:]
    train_label = label[test_set_size:]
    """test_data = torch.Tensor(data[:test_set_size])
    test_label=torch.Tensor(label[:test_set_size])
    train_data = torch.Tensor(data[test_set_size:])
    train_label=torch.Tensor(label[test_set_size:])"""
    # iloc选择参数序列中所对应的行
    return train_data, train_label, test_data, test_label


traindata, trainlabel, testdata, testlabel = split_train_test(data, label, 0.2)
from torch.utils.data import Dataset, DataLoader, TensorDataset

"""
traindata = TensorDataset(traindata, trainlabel)
testdata = TensorDataset(testdata, testlabel)"""
EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
TIME_STEP = 28  # rnn time step / image height
INPUT_SIZE = 28  # rnn input size / image width
LR = 0.001
print(len(testlabel))

def collate(samples):
    #print("samples: ", type(samples))
    #print("samples shape: ", len(samples), len(samples[0]), samples[0][-1])
    ##  zip 打包数据 ；  zip(*) zip的逆操作
    ##  map(a,b) 将b中每个元素转化成a类（进行操作）
    ##  将samples转置并拆解 samples: batch_size * 2400
    graphs, labels = map(list, zip(*samples))
    #print("graphs: ", graphs, ". labels: ", labels)
    # dgl.batch 将一批图看作是具有许多互不连接的组件构成的大型图
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)


rnn = Classifier(150//depth, 512, 21).to(device)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all parameters
loss_func = nn.CrossEntropyLoss()
"""print(len(traindata))
print(traindata[-1])
print(len(testdata))
print(testdata[-1])"""
trainset = RFDataset(traindata, trainlabel)
testset = RFDataset(testdata, testlabel)
""""""
def get_confusion_matrix(preds, labels, num_classes, normalize="true"):
    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    # If labels are one-hot encoded, get their indices.
    if labels.ndim == preds.ndim:
        labels = torch.argmax(labels, dim=-1)
    # Get the predicted class indices for examples.
    preds = torch.flatten(torch.argmax(preds, dim=-1))
    labels = torch.flatten(labels)
    cmtx = confusion_matrix(labels, preds, labels=list(range(num_classes)))#, normalize=normalize) 部分版本无该参数
    return cmtx


train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
# print(type(train_loader))
def train(EPOCH):
    #print("0train")
    #for epoch in range(EPOCH):
        #print("train")
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
        #print(b_x)
        # b_x = RFDataset.create_graph(cg,b_x)
        # b_x = RFDataset.create_graph(b_x)
        b_y = torch.tensor(b_y, dtype=torch.int64)
        b_x, b_y = b_x.to(device), b_y.to(device)
        #b_x = b_x.view(-1, 1, 2400)
        # print(b_y)
        output = rnn(b_x)  # rnn output
        # print(output.dtype)
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        '''if step % 50 == 0:
            test_output = rnn(b_x)  # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)'''

acclist=[]
cmtxlist=[]
def test(ep):
    test_loss = 0
    correct = 0
    preds = []
    labels = []
    best_acc = 0
    # data, label = data0,label0
    #print("test")

    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    if ep == 1:
        torch.save(rnn.state_dict(), 'Poolh.mdl')
    """"""
    with torch.no_grad():
        # print(type(test_data))
        for data, target in test_loader:
            # data = RFDataset.create_graph(cg, data)
            target = torch.tensor(target, dtype=torch.int64)
            # data, target = data.to(device), target.to(device)
            #data = data.view(-1, 1, 2400)
            data, target = data.to(device), target.to(device)
            #data, target = Variable(data, volatile=True), Variable(target)
            output = rnn(data).to(device)

            preds.append(output.cpu())
            labels.append(target.cpu())
            test_loss += loss_func(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            """"""
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        cmtx = get_confusion_matrix(preds, labels, 21)
        print(cmtx)
        for i in range(len(cmtx)):
            print(cmtx[i][i]/sum(cmtx[i]))
        """"""
        # print('============================')
        # print(len(test_data.dataset))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        acc = correct / len(test_loader.dataset)
        acclist.append(float(acc))
        cmtxlist.append(cmtx.tolist())
        if ep == 54:
            with open("RNN.txt", "a+") as f:
                #f.write('\n时空骨架层数: {}, batch size: {}, 输入、隐藏层{}、{}\n'.format(depth,BATCH_SIZE,150//depth,512))
                #f.write('\n')
                f.write('\n')
                f.write(str(acclist))
                f.write('\n')
                f.write(str(max(acclist)))
                f.write(str(cmtxlist[acclist.index(max(acclist))]))
                f.write('\n')
        print(max(acclist))
        return test_loss

import time
if __name__ == '__main__':
    for epoch in range(0, 55):
        print(epoch)
        Stime=time.time()
        train(epoch)
        test(epoch)
        Etime=time.time()
        print("ctime=",Etime-Stime)
        if epoch % 20 == 0:
            LR /= 10
