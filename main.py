#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 19:56:25 2019

@author: vijay
"""


import torch
from skimage import io
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import glob
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.transform import warp, AffineTransform, resize


trdata = "./data
tstdata = "./data
#Preparing training and test data pairs/ moving and fixed images to input into the registration framework.    
train_x = []
for img_train in glob.glob(trdata):
    n = io.imread(img_train)
    n = rgb2gray(n)
    n= resize(n,(28,28))
    train_x.append(n.reshape(1, 28, 28))
    
    
train_y = []
for img_test in glob.glob(tstdata):
    n = io.imread(img_test)
    n= resize(n,(28,28))
    n = rgb2gray(n)
    train_y.append(n.reshape(1, 28, 28))
    


#Sample test image to check the visualization

test_x = io.imread('path-to-test-data')  
test_x= resize(test_x,(28,28))
test_x = rgb2gray(test_x)
test_x = test_x.reshape(1, 1, 28, 28) 

test_y = io.imread('path-to-ground-truth')
test_y= resize(test_y,(28,28))
test_y = rgb2gray(test_y)
test_y = test_y.reshape(1, 1, 28, 28)




#Model to Align the image : this model contains spatial transformer module with decoder - encoder architecture







class MultiReg(nn.Module):
     
    def __init__(self):
        super(MultiReg, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(10, 8, 3, stride=2, padding=3),  # b, 64, 15, 15
            nn.ReLU(True),
            nn.Conv2d(8, 3, 3, stride=2, padding=1),  # b, 64, 15, 15
            nn.ReLU(True),
            nn.Conv2d(3, 2, 3, stride=2, padding=1),  # b, 64, 15, 15
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 3, 3, stride=2, padding =1),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(3, 8, 3, stride=2, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 10, 3, stride=2, padding=3),  # b, 1, 28, 28
            nn.Tanh()
        )
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        #7926 parameters.
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x, moving):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(moving, grid)
        
        return x, grid
   
    def forward(self, x, moving):
       # x = self.stn(x)
       
        x = self.decoder(x)
        x = self.encoder(x)              
        
        x, grid = self.stn(x, moving)
        
        return x, grid
    
    
    
    

   
model = MultiReg().cuda()
print(sum(p.numel() for p in model.parameters() if p.requires_grad))


#Loss function to check the accuracy of the model
#NCC LOSS and DIce loss
def NCC(A, B):
    #Find the normalized cross correlation between two matrices
    #Using pytorch libraries.
    A_submean = A - A.mean()
    B_submean = B - B.mean()
    
    Numerator = torch.sum( A_submean * B_submean )
    Denominator = torch.sqrt ( torch.sum( torch.mul(A_submean, A_submean) ) * torch.sum( torch.mul(B_submean, B_submean) ))
    Ncc = Numerator/Denominator
    return Ncc


def dice_loss(pred, target):



    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )








def show_plot(iteration,loss):
    
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.plot(iteration,loss)
    plt.show()


num_epochs = 100
batch_size = 1
learning_rate = 1e-3

losslist = []
trainepoch = []

#Loss function used in the model is mean square error loss
# =============================================================================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
# =============================================================================
for epoch in range(num_epochs):
    for i in range(len(train_x)):
        fixed = torch.tensor(train_x[i])
        fixed = fixed.type(torch.FloatTensor)
        fixed = Variable(fixed).cuda()
        fixed = fixed.reshape(1, 1, 28, 28)
        moving = torch.tensor(train_y[i])
        moving = moving.type(torch.FloatTensor)
        moving = Variable(moving).cuda()
        moving = moving.reshape(1, 1, 28, 28)
        fixed_moving = torch.cat((fixed, moving), 1)   # 1 , 2, 28, 28
        
        
        output, grid = model(fixed_moving, moving)
        #output = Variable(output).cuda()
        loss = criterion(output, fixed)
        #loss = criterion(output, moving)
        loss.requires_grad_(True)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data[0]))
#show_plot(trainepoch,loss)
        
        
        
#DVF visualization
import numpy as np
def dvf_Visualization(grid):
    n = 28
    X, Y = np.mgrid[0:28, 0:28]
    Z = grid.cpu()
    z = Z.detach()
    z = z.numpy().reshape(2, 28, 28)
    U = z[0]
    V = z[1]
    #plt.axes([0.025, 0.025, 0.95, 0.95])
    
    plt.quiver(X, Y, U, V,  linewidth=.1)
    plt.xlim(-1, n)
    plt.xticks(())
    plt.ylim(-1, n)
    plt.yticks(())
    plt.show()


fixed = torch.tensor(test_x)
fixed = fixed.type(torch.FloatTensor)
fixed = Variable(fixed).cuda()
fixed = fixed.reshape(1, 1, 28, 28)

moving = torch.tensor(test_y)
moving = moving.type(torch.FloatTensor)
moving = Variable(moving).cuda()
moving = moving.reshape(1, 1, 28, 28)

fixed_moving = torch.cat((fixed, moving), 1)
result_image, grid = model(fixed_moving, moving)
dvf_Visualization(grid)

result_image1 = result_image.reshape(28,28)
r = Variable(result_image1).cpu().detach().numpy()




fig, axes = plt.subplots(nrows=1, ncols=3)

ax = axes.ravel()

ax[0].imshow(test_x.reshape(28,28), cmap='gray')


fig, axes = plt.subplots(nrows=1, ncols=3)
ax[0].set_title("Reference Image")

ax[2].imshow(r, cmap='gray')
ax[2].set_title("Registered Image")

ax[1].imshow(test_y.reshape(28,28), cmap='gray')
ax[1].set_title("Moving Image")


from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ax = axes.ravel()

plt.tight_layout()
plt.show()



print("NCC: ",NCC(fixed,result_image1))
 
    
    
    