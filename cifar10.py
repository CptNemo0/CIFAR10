import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import argparse

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(3, 32, 3, 1)
        self.conv2=nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.lin1 = nn.Linear(12544, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3 = nn.Linear(512, 256)
        self.lin4 = nn.Linear(256, 128)
        self.lin5 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.lin4(x)
        x = F.relu(x)
        x = self.lin5(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, epoch, trainLoader, device, optimizer, loss_function):
    model.train()
    print("Epoch: {}".format(epoch))

    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

def validation(model, testLoader, device, loss_funciton):
    model.eval()
    test_loss = 0
    correct = 0
    amount = len(testLoader.dataset)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(testLoader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_funciton(output, target)
            test_loss = test_loss + loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = test_loss / amount

    print("\nValidation set:\n\tLoss: {:.5f}, Accuracy : {:.5f}. \n".format(test_loss, 100.0 * correct / amount))
   
def main():

    parser = argparse.ArgumentParser(description="PyTorch CIFAR10")
    parser.add_argument('--trainingBatchSize', type=int, default=128)
    parser.add_argument('--validationBatchSize', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learningRate', type=float, default=4e-3)
    parser.add_argument('--momentum', type=float, default=9e-1)
    parser.add_argument('--weightDecay', type=float, default=1e-3)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net()
    net.to(device)
    optimizer = optim.SGD(net.parameters() ,lr=args.learningRate, momentum=args.momentum, weight_decay=args.weightDecay)#args
    loss_function = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    training_transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                             transforms.ToTensor(), 
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    validation_transform = transforms.Compose([transforms.ToTensor(), 
                                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    training_set = torchvision.datasets.CIFAR10(root="./CIFAR10Dataset", train=True, download=True, transform=training_transform)
    validation_set = torchvision.datasets.CIFAR10(root="./CIFAR10Dataset", train=False, download=True, transform=validation_transform)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.trainingBatchSize, shuffle=True, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.validationBatchSize, shuffle=False, num_workers=4)

    for epoch in range(0, args.epochs):
        train(net, epoch, train_loader, device, optimizer, loss_function)
        validation(net, validation_loader, device, loss_function)
        scheduler.step()

if __name__ == '__main__':
    main()

