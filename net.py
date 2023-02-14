


import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x,dim=1)


if __name__ == "__main__":
    import torch
    import torchvision
    from torchvision import transforms, datasets

    train = datasets.MNIST('', train = True, download = True, transform=transforms.Compose([transforms.ToTensor()]))

    test = datasets.MNIST('', train = False, download = True, transform=transforms.Compose([transforms.ToTensor()]))

    trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle=True)
    trainset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle=False)




    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    EPOCHS = 40

    for e in range(EPOCHS):
        for data in trainset:
            # Features, labels
            x,y = data
            net.zero_grad()
            out = net(x.view(-1,28*28))
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
        print(loss)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainset:
            x, y = data
            out = net(x.view(-1, 28*28))
            for idx, i in enumerate(out):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total +=1
    print("Accuracy Score:",correct/total)

    torch.save(net, 'net.pth')