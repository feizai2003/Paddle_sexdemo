import paddle
from paddle.io import Dataset
import paddle.nn as nn
import numpy as np

paddle.device.set_device('cpu')
# 模型构建
class SexNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(2,2),
            # nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


# 数据读取
class MyDataset(Dataset):
    def __init__(self,label_path,transform=None):
        super(MyDataset,self).__init__()
        self.data_list = []
        with open(label_path,'r',encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').rstrip().split()
                self.data_list.append([np.float32(line[0]),np.float32(line[1]),int(line[2])])
        self.transform = transform

    def __getitem__(self,index):
        data = self.data_list[index]
        return paddle.Tensor(np.array([data[0],data[1]])),paddle.Tensor(np.array(data[2]))

    def __len__(self):
        return len(self.data_list)
# 训练
def train():
    batch_size = 8
    train_data = MyDataset(label_path='sex_train.txt')
    val_data = MyDataset(label_path='sex_val.txt')

    train_loader = paddle.io.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
    val_loader = paddle.io.DataLoader(dataset=val_data,batch_size=batch_size)
    model = SexNet()
    optim = paddle.optimizer.Adam(learning_rate=0.01,weight_decay=1e-3,parameters=model.parameters())

    loss_fn = paddle.nn.CrossEntropyLoss()
    epochs = 100
    for epoch in range(epochs):
        acc = 0
        loss = 0
        max_acc = 0
        for batch_id,(feature,label) in enumerate(train_loader):
            out = model(feature)
            loss = loss_fn(out, label)
            label = label.unsqueeze(1)
            acc = paddle.metric.accuracy(out,label)
            loss.backward()

            print("epoch: {},batch_id: {}, loss is： {}, acc is:{}".format(
                epoch,batch_id+1,loss.numpy(),acc.numpy()
            ))

            optim.step()
            optim.clear_grad()

        for batch_id ,(feature,label) in enumerate(val_loader):
            predicts = model(feature)
            loss = loss_fn(predicts, label)
            label = label.unsqueeze(1)
            acc = paddle.metric.accuracy(predicts,label)

            print("batch_id: {}, loss is :{}, acc is: {}".format(
                batch_id+1,loss.numpy(),acc.numpy()
            ))

        if acc > max_acc:
            paddle.save(model.state_dict(),"best_model.pdparams")
def init_weight():
    layer_state_dict = paddle.load("best_model.pdparams")
    layer = SexNet()
    layer.set_state_dict(layer_state_dict)
    return layer

def predict(layer,input1, input2):
    input = paddle.Tensor(np.array([input1, input2]))
    output = layer(input)
    output = paddle.argmax(output)
    return output.numpy()

if __name__ == '__main__':
    train()
    print("finished")