from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
import torch
import numpy as np
import random
import torch
import sys

train_data_path = sys.argv[1]
model_save_path = sys.argv[2]

def preprocess(image_list):
    """ Normalize Image and Permute (N,H,W,C) to (N,C,H,W)
    Args:
      image_list: List of images (9000, 32, 32, 3)
    Returns:
      image_list: List of images (9000, 3, 32, 32)
    """
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = (image_list / 255.0) * 2 - 1
    image_list = image_list.astype(np.float32)
    return image_list

class Image_Dataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        images = self.image_list[idx]
        return images

## Reading data
trainX = np.load(train_data_path)
trainX_preprocessed = preprocess(trainX)
img_dataset = Image_Dataset(trainX_preprocessed)


def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

## Model
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(len(input), -1)
    
class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(-1, 1024, 1, 1)

## VAE
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        
        self.flatten = Flatten()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.unflatten = UnFlatten()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, 17, stride=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
        
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        
        z, mu, logvar = self.bottleneck(x)
        z = self.fc3(z)
        
        tmp = self.unflatten(z)
        
        output  = self.decoder(tmp)
        return z, output, mu, logvar

## Training
same_seeds(0)

model = VAE().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
model.train()
n_epoch = 200

# 準備 dataloader, model, loss criterion 和 optimizer
img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)

# 主要的訓練過程
epoch_loss = 0
for epoch in range(n_epoch):
    epoch_loss = 0
    for data in img_dataloader:
        
        ## Training
        img = data
        img = img.cuda()

        z, output, mu, logvar = model(img)
        loss = criterion(output, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, epoch_loss))
# 訓練完成後儲存 model
torch.save(model.state_dict(), model_save_path)

print("Finished")