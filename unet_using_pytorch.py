#!/usr/bin/env python
# coding: utf-8

# # UNET IMPLEMENTATION USING PYTORCH

# #### Importing reqired libraries

# In[59]:


import torch
import torch.nn as nn


# ### The UNET architecture:

# ![u-net-architecture.png](attachment:u-net-architecture.png)

# ## It consists of: 
# ### Contracting path (ENCODER)
# (Which is like a typical Convolutioal Neural Network architecture)
# and 
# ### Expanding path (DECODER)

# > **in_c and out_c** are the input and output channels parameters 
# 
# > **inplace=True** modifies the input directly
# 
# > **torch.nn.Modules** is the base class for nn modules
# 

# The U-net downscaling path architecture consists of the repeated application of-
# * (3x3) convolutional layers
# * ReLU layer 
# 
# Followed by a Max pooling layer (with stride=2)

# The U-net upscaling path architecture consists of the repeated application of-
# * (3x3) convolutional layers
# * ReLU layer 
# 
# Followed by tranposed convolutional layer

# #### Double convolutional layer:

# In[70]:


def conv_repeat(in_c,out_c): 
    cnn=nn.Sequential(
        
        nn.Conv2d(in_c,out_c,kernel_size=3),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(out_c,out_c,kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return cnn


# > xone is the output of down_conv_1

# We define another function crop() to crop the images that need to be copied while upsampling

# In[71]:


def crop(tensor,target_tensor):
    target_size=target_tensor.size()[2]
    tensor_size=tensor.size()[2]
    
    delta=tensor_size-target_size
    delta=delta//2
    
    return tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]
    # tensor[batch_size, channel, height, width]


# #### Unet architecture:

# In[72]:


class UNET(nn.Module): 
    
    def __init__(self):
        super(UNET,self).__init__()
        
        self.max_pool_2x2= nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.down_conv_1=conv_repeat(1,64)
        self.down_conv_2=conv_repeat(64,128)
        self.down_conv_3=conv_repeat(128,256)
        self.down_conv_4=conv_repeat(256,512)
        self.down_conv_5=conv_repeat(512,1024)
        
        
        
        self.up_transpose_1=nn.ConvTranspose2d(in_channels=1024,
                                         out_channels=512,
                                         kernel_size=2,
                                         stride=2)
        
        self.up_conv_1=conv_repeat(1024,512)
        
        
        self.up_transpose_2=nn.ConvTranspose2d(in_channels=512,
                                         out_channels=256,
                                         kernel_size=2,
                                         stride=2)
        
        self.up_conv_2=conv_repeat(512,256)
        
        
        self.up_transpose_3=nn.ConvTranspose2d(in_channels=256,
                                         out_channels=128,
                                         kernel_size=2,
                                         stride=2)
        
        self.up_conv_3=conv_repeat(256,128)
        
        
        self.up_transpose_4=nn.ConvTranspose2d(in_channels=128,
                                         out_channels=64,
                                         kernel_size=2,
                                         stride=2)
        
        self.up_conv_4=conv_repeat(128,64)
        
        
        
        self.output= nn.Conv2d(in_channels=64,
                              out_channels=2,
                              kernel_size=1)
        
        
    
    def forward(self,img):
        
        #Contracting path
        xone=self.down_conv_1(img)
        
#         print(xone.size())
        xtwo=self.max_pool_2x2(xone)
        
        xthree=self.down_conv_2(xtwo)
        xfour=self.max_pool_2x2(xthree)
        
        xfive=self.down_conv_3(xfour)
        xsix=self.max_pool_2x2(xfive)
        
        xseven=self.down_conv_4(xsix)
        xeight=self.max_pool_2x2(xseven)
        
        xnine=self.down_conv_5(xeight)
        
#         print(xnine.size())
        
        #Expanding path
        x=self.up_transpose_1(xnine)
        xcrop=crop(xseven,x)
        x=self.up_conv_1(torch.cat([x,xcrop],1))
        
        x=self.up_transpose_2(x)
        xcrop=crop(xfive,x)
        x=self.up_conv_2(torch.cat([x,xcrop],1))
        
        x=self.up_transpose_3(x)
        xcrop=crop(xthree,x)
        x=self.up_conv_3(torch.cat([x,xcrop],1))
        
        x=self.up_transpose_4(x)
        xcrop=crop(xone,x)
        x=self.up_conv_4(torch.cat([x,xcrop],1))
        
        x=self.output(x)
        print(x.size())
        
        return x
        
#         print("Uncropped ",xseven.size())
#         print("Cropped: ",xcrop.size())


# In[73]:


if __name__=="__main__":
    img=torch.rand((1,1,572,572)) 
    #(batch_size,channel,height,width)
    model=UNET()
    print(model(img))


# In[ ]:





# In[ ]:




