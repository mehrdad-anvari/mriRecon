import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from torchsummary import summary
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, layer):
        device = layer.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = layer[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class u_net_bn(nn.Module):
    def __init__(self,ngpu):
        super(u_net_bn, self).__init__()
        self.ngpu = ngpu
        self.df_dim = 64
        self.conv1 = nn.Conv2d(2, self.df_dim, (4, 4), (2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(1 * self.df_dim, 2 * self.df_dim, (4, 4), (2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(2 * self.df_dim, 4 * self.df_dim, (4, 4), (2, 2), padding=(1, 1))
        self.conv4 = nn.Conv2d(4 * self.df_dim, 8 * self.df_dim, (4, 4), (2, 2), padding=(1, 1))
        self.conv5 = nn.Conv2d(8 * self.df_dim, 8 * self.df_dim, (4, 4), (2, 2), padding=(1, 1))
        self.conv6 = nn.Conv2d(8 * self.df_dim, 8 * self.df_dim, (4, 4), (2, 2), padding=(1, 1))
        self.conv7 = nn.Conv2d(8 * self.df_dim, 8 * self.df_dim, (4, 4), (2, 2), padding=(1, 1))
        self.conv8 = nn.Conv2d(8 * self.df_dim, 8 * self.df_dim, (2, 2), (2, 2))
        self.up7 = nn.ConvTranspose2d(8 * self.df_dim, 8 * self.df_dim, (2, 2), stride=(2, 2), padding=(0, 0))
        self.up6 = nn.ConvTranspose2d(16 * self.df_dim, 16 * self.df_dim, (4, 4), (2, 2), padding=(1, 1))
        self.up5 = nn.ConvTranspose2d(24 * self.df_dim, 16 * self.df_dim, (4, 4), (2, 2), padding=(1, 1))
        self.up4 = nn.ConvTranspose2d(24 * self.df_dim, 16 * self.df_dim, (4, 4), (2, 2), padding=(1, 1))
        self.up3 = nn.ConvTranspose2d(24 * self.df_dim, 4 * self.df_dim, (4, 4), (2, 2), padding=(1, 1))
        self.up2 = nn.ConvTranspose2d(8 * self.df_dim, 2 * self.df_dim, (4, 4), (2, 2), padding=(1, 1))
        self.up1 = nn.ConvTranspose2d(4 * self.df_dim, 1 * self.df_dim, (4, 4), (2, 2), padding=(1, 1))
        self.up0 = nn.ConvTranspose2d(2 * self.df_dim, 1 * self.df_dim, (4, 4), (2, 2), padding=(1, 1))

        self.batch_conv2 = nn.BatchNorm2d(2 * self.df_dim)
        self.batch_conv3 = nn.BatchNorm2d(4 * self.df_dim)
        self.batch_conv4 = nn.BatchNorm2d(8 * self.df_dim)
        self.batch_conv5 = nn.BatchNorm2d(8 * self.df_dim)
        self.batch_conv6 = nn.BatchNorm2d(8 * self.df_dim)
        self.batch_conv7 = nn.BatchNorm2d(8 * self.df_dim)
        self.batch_up0 = nn.BatchNorm2d(1 * self.df_dim)
        self.batch_up1 = nn.BatchNorm2d(1 * self.df_dim)
        self.batch_up2 = nn.BatchNorm2d(2 * self.df_dim)
        self.batch_up3 = nn.BatchNorm2d(4 * self.df_dim)
        self.batch_up4 = nn.BatchNorm2d(16 * self.df_dim)
        self.batch_up5 = nn.BatchNorm2d(16 * self.df_dim)
        self.batch_up6 = nn.BatchNorm2d(16 * self.df_dim)
        self.batch_up7 = nn.BatchNorm2d(8 * self.df_dim)

        self.outLayer = nn.Conv2d(self.df_dim, 1, (1, 1))

    def forward(self, x):
        x = self.conv1(x)
        c1 = nn.LeakyReLU(0.2)(x)

        
        x = self.conv2(x)
        x = self.batch_conv2(x)
        c2 = nn.LeakyReLU(0.2)(x)

        
        x = self.conv3(c2)
        x = self.batch_conv3(x)
        c3 = nn.LeakyReLU(0.2)(x)

        
        x = self.conv4(x)
        x = self.batch_conv4(x)
        c4 = nn.LeakyReLU(0.2)(x)

        
        x = self.conv5(c4)
        x = self.batch_conv5(x)
        c5 = nn.LeakyReLU(0.2)(x)

        
        x = self.conv6(c5)
        x = self.batch_conv6(x)
        c6 = nn.LeakyReLU(0.2)(x)

        c7 = self.conv7(c6)
        c7 = self.batch_conv7(c7)

        x = nn.LeakyReLU(0.2)(c7)
        x = self.conv8(x)
       
        u7 = self.up7(x)
        u7 = self.batch_up7(u7)
        u7 = nn.ReLU()(u7)

        x = torch.concat((u7, c7), dim=1)
        x = self.up6(x)
        u6 = self.batch_up6(x)
        u6 = nn.ReLU()(u6)

        x = torch.concat((u6, c6), dim=1)
        x = self.up5(x)
        x = self.batch_up5(x)
        u5 = nn.ReLU()(x)

    
        x = torch.concat((u5, c5), dim=1)
        x = self.up4(x)
        x = self.batch_up4(x)
        u4 = nn.ReLU()(x)

        
        x = torch.concat((u4, c4), dim=1)
        x = self.up3(x)
        x = self.batch_up3(x)
        u3 = nn.ReLU()(x)

        
        x = torch.concat((u3, c3), dim=1)
        x = self.up2(x)
        x = self.batch_up2(x)
        u2 = nn.ReLU()(x)
        
        x = torch.concat((u2, c2), dim=1)
        x = self.up1(x)
        x = self.batch_up1(x)
        u1 = nn.ReLU()(x)
        
        x = torch.concat((u1, c1), dim=1)
        x = self.up0(x)
        x = self.batch_up0(x)
        p2d = (0, 1, 0, 1) 
        x = F.pad(x, p2d, "constant", 0)
        x = nn.ReLU()(x)

        x = self.outLayer(x)
        x = nn.Tanh()(x)
        return x


class KspaceNetT1(nn.Module):
    def __init__(self,ngpu):
        super(KspaceNetT1, self).__init__()
        self.df_dim = 32     
        self.posEn_dim = 34
        self.layEn_dim = 32
        self.conv1 = nn.Conv2d(10 + self.posEn_dim + self.layEn_dim , 4*self.df_dim, (1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(4*self.df_dim, 2*self.df_dim, (1, 1), padding=(0, 0))
        self.conv3 = nn.Conv2d(2*self.df_dim, 1*self.df_dim, (1, 1), padding=(0, 0))
        self.conv4 = nn.Conv2d(1*self.df_dim, 2, (1, 1), padding=(0, 0))

        self.batch1 = nn.BatchNorm2d(4 * self.df_dim)
        self.batch2 = nn.BatchNorm2d(2 * self.df_dim)
        self.batch3 = nn.BatchNorm2d(1 * self.df_dim)

        self.layer_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.layEn_dim),
            nn.Linear(self.layEn_dim, self.layEn_dim),
            nn.GELU(),
            nn.Linear(self.layEn_dim, self.layEn_dim),
        )
        
        self.layer_mlp2 = nn.Sequential(
            SinusoidalPositionEmbeddings(self.layEn_dim),
            nn.Linear(self.layEn_dim, self.layEn_dim),
            nn.GELU(),
            nn.Linear(self.layEn_dim, 2*self.posEn_dim),
        )
        
        self.layer_mlp3 = nn.Sequential(
            SinusoidalPositionEmbeddings(self.layEn_dim),
            nn.Linear(self.layEn_dim, self.layEn_dim),
            nn.GELU(),
            nn.Linear(self.layEn_dim, 10),
        )
        
        self.convLEPEforX = nn.Conv2d(self.posEn_dim + 10, 10*2, (1, 1), padding=(0, 0))
        
    def forward(self, x, PE, LE):

        layerEmbeddingLatent = self.layer_mlp(LE)
        
        LEchs = rearrange(layerEmbeddingLatent, "b c -> b c 1 1")
        LEchs = LEchs.repeat(1,1,257,257) 
        LEforPE = self.layer_mlp2(LE)
        LEforPE2 = rearrange(LEforPE, "b c -> b c 1 1")
        scale_shift = LEforPE2.chunk(2, dim=1)
        scale, shift = scale_shift
        PE = PE * (scale + 1) + shift
        LEforX = self.layer_mlp3(LE)
        LEforX2 = rearrange(LEforX, "b c -> b c 1 1")
        LEforX2 = LEforX2.repeat(1,1,257,257)
        LEPEforX = torch.concat((PE, LEforX2),dim=1)
        LEPEforX2 = self.convLEPEforX(LEPEforX)
        scale_shift = LEPEforX2.chunk(2, dim=1)
        scale, shift = scale_shift
        x = x * (scale + 1) + shift
                
        x = torch.concat((x, PE, LEchs),dim=1)

        x = self.conv1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.batch1(x)

        x = self.conv2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.batch2(x)

        x = self.conv3(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.batch3(x)

        x = self.conv4(x)
        
        return x
    

netG = u_net_bn(1).to(device)
summary(netG, input_size=(2, 257, 257))
