import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20*10*10, 500)
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        input_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(input_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

def train(net, device, train_data, train_labels, optimizer, epoch, batch_size, test_data, test_labels):
    for i in range(epoch):
        train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
        final_loss = 0
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)
            true_labels = F.one_hot(labels, 10).float()
            prob = net(features)
            loss = nn.CrossEntropyLoss()(prob, true_labels)
            final_loss = loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        test_index = torch.randint(0, test_data.size(0), [1000])
        accuracy = mytest(net, device, test_data[test_index], test_labels[test_index])
        print("epoch: %d, loss: %.6f, accuracy: %.4f" % (i+1, final_loss, accuracy))

def mytest(net, device, test_data, test_labels):
    corr = 0
    total = test_data.size(0)
    with torch.no_grad():
        for i in range(total):
            feature, label = test_data[i], test_labels[i]
            feature, label = feature.to(device), label.to(device)
            prob = net(feature)
            index = Categorical(prob).sample()
            if index == label:
                corr += 1
    return corr / total

if __name__ == "__main__":
    train_data = torch.load('train_data.pth')
    train_labels = torch.load('train_labels.pth')
    test_data = torch.load('test_data.pth')
    test_labels = torch.load('test_labels.pth')
    # mode = 'train_and_test'  # 如果还没有训练出模型，选这个模式
    mode = 'load_and_test'  # 如果已经有训练好的模型，选这个模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义网络. `MyConvNet`类对象构造时不需要传入参数以方便批改作业
    net = MyConvNet().to(device)
    if mode == 'train_and_test':
        batch_size = 400
        epochs = 20
        lr = 0.005
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=0.01)
        train(net, device, train_data, train_labels, optimizer, epochs, batch_size, test_data, test_labels)
        # torch.save(net.state_dict(), 'mymodel.pth')
    # 如果已经训练完成, 则直接读取网络参数. 注意文件名改为自己的信息
    elif mode == 'load_and_test':
        net.load_state_dict(torch.load('mymodel.pth'))
    accuracy = mytest(net, device, test_data, test_labels)
    print("accuracy: %.4f" % accuracy)
