import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

from PIL import Image

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  
        self.conv1=nn.Conv2d(3, 32, 3, 1)
        self.conv2=nn.Conv2d(32, 64, 3, 1)
        self.conv3=nn.Conv2d(64, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.4)
        self.lin1 = nn.Linear(10816, 4096)
        self.lin2 = nn.Linear(4096, 2048)
        self.lin3 = nn.Linear(2048, 1024)
        self.lin4 = nn.Linear(1024, 512)
        self.lin5 = nn.Linear(512, 256)
        self.lin6 = nn.Linear(256, 128)
        self.lin7 = nn.Linear(128, 64)
        self.lin8 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
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
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.lin6(x)
        x = F.relu(x)
        x = self.lin7(x)
        x = F.relu(x)
        x = self.lin8(x)
        output = F.log_softmax(x, dim=1)
        return output

def main():
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 

    net = Net()
    prmtrsPath = "Path to parameters"   
    net.load_state_dict(torch.load(prmtrsPath))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    net.eval()
    
    imgPath = "Path to .png image"            
    img = Image.open(imgPath)
    preprocess = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  
    img_t = preprocess(img)
    img_t = img_t.to(device)

    batch_t = torch.unsqueeze(img_t, 0)                      
    out = net(batch_t)    
    another, index = torch.max(out, 1)
    print("Oh! It's a " + str(classes[index])+ "!!! ")
                                    
if __name__ == '__main__':
    main()