import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

### UNet model      
class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_features=6, 
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
    ):
        """
        Adapted U-Net for regression to predict landmarks coordinates.
        Adapted from https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py

        Args:
            in_channels (int): number of input channels.
            out_features (int): number of output features (coordinates).
            depth (int): depth of the network.
            wf (int): number of filters in the first layer is 2**wf.
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output. This may introduce artifacts.
            batch_norm (bool): Use BatchNorm after layers with an activation function.
            up_mode (str): one of 'upconv' or 'upsample'.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, out_features, kernel_size=1)
        self.fc = nn.Linear(out_features * 16 * 16, out_features)  # Assuming output size is at least 16x16

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != self.depth - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        x = self.last(x)
        x = F.adaptive_avg_pool2d(x, (16, 16))  # Pool to a fixed size
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_with_blocks(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != self.depth - 1:  # Save feature maps before downsampling
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        # Upsampling path
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])  # Use the saved feature maps

        # Final convolutions (may be adjusted based on your specific design)
        x = self.last(x)
        x = F.adaptive_avg_pool2d(x, (16, 16))  # Pool to a fixed size
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, blocks  # Return both the final output and the intermediate blocks


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def forward(self, x, bridge):
        up = self.up(x)
        # Calculate padding
        diffY = torch.tensor([bridge.size()[2] - up.size()[2]])
        diffX = torch.tensor([bridge.size()[3] - up.size()[3]])
        # Pad the upsampled feature map to have the same size as the bridge feature map
        up = F.pad(up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out



###############

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


    
#####
class AddCoordinates(nn.Module):
    """Layer to add X,Y coordinates as channels at runtime."""
    def forward(self, input_tensor):
        batch_size, _, height, width = input_tensor.size()
        # Transform coordinates to range from 0 to 1
        x_coords = torch.linspace(0, 1, width, device=input_tensor.device).repeat(batch_size, height, 1)
        y_coords = torch.linspace(0, 1, height, device=input_tensor.device).repeat(batch_size, width, 1).transpose(1, 2)
        coords = torch.stack((x_coords, y_coords), dim=1)
        return torch.cat((input_tensor, coords), dim=1)


class CoordConv(nn.Module):
    """CoordConv layer replacing the initial Conv2d in ConvBlock."""
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CoordConv, self).__init__()
        self.add_coords = AddCoordinates()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.add_coords(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

##### 

#### Attention Guided Regression UNet
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super(AttentionBlock, self).__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out

class RAUNet(nn.Module):
    """
    Adapted from https://github.com/sfczekalski/attention_unet/blob/master/model.py
    Regression Attention UNet for landmark detection.
    
    It uses attention gates to refine the feature maps at each level of the U-Net.
    Predicts the coordinates of the landmarks.
    
    Args:
        in_channels (int): number of input channels.
        out_features (int): number of output features (coordinates).
    """
    def __init__(self, in_channels=1, out_features=6):
        super(RAUNet, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.Conv1 = ConvBlock(in_channels, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        # Decoder
        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        # Final layers for regression
        self.last = nn.Conv2d(64, out_features, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(out_features * 16 * 16, out_features)  # Assuming output size is at least 16x16

    def forward(self, x):
        # Encoder
        e1 = self.Conv1(x)
        e2 = self.MaxPool(e1); e2 = self.Conv2(e2)
        e3 = self.MaxPool(e2); e3 = self.Conv3(e3)
        e4 = self.MaxPool(e3); e4 = self.Conv4(e4)
        e5 = self.MaxPool(e4); e5 = self.Conv5(e5)

        # Decoder
        d5 = self.Up5(e5); s4 = self.Att5(gate=d5, skip_connection=e4); d5 = torch.cat((s4, d5), dim=1); d5 = self.UpConv5(d5)
        d4 = self.Up4(d5); s3 = self.Att4(gate=d4, skip_connection=e3); d4 = torch.cat((s3, d4), dim=1); d4 = self.UpConv4(d4)
        d3 = self.Up3(d4); s2 = self.Att3(gate=d3, skip_connection=e2); d3 = torch.cat((s2, d3), dim=1); d3 = self.UpConv3(d3)
        d2 = self.Up2(d3); d2 = torch.cat((e1, d2), dim=1); d2 = self.UpConv2(d2)

        # Regression output
        out = self.last(d2)
        out = F.adaptive_avg_pool2d(out, (16, 16))  # Pool to a fixed size
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    

class CRAUNet(nn.Module):
    """
    Adapted from https://github.com/sfczekalski/attention_unet/blob/master/model.py
    Coordinate-aware Attention UNet for landmark detection.
    
    It uses attention gates to refine the feature maps at each level of the U-Net.
    Predicts the coordinates of the landmarks.
    
    Args:
        in_channels (int): number of input channels.
        out_features (int): number of output features (coordinates).
    """
    def __init__(self, in_channels=1, out_features=6):
        super(CRAUNet, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.Conv1 = CoordConv(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        # Decoder
        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        # Final layers for regression
        self.last = nn.Conv2d(64, out_features, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(out_features * 16 * 16, out_features)  # Assuming output size is at least 16x16

    def forward(self, x):
        # Encoder
        e1 = self.Conv1(x)
        e2 = self.MaxPool(e1); e2 = self.Conv2(e2)
        e3 = self.MaxPool(e2); e3 = self.Conv3(e3)
        e4 = self.MaxPool(e3); e4 = self.Conv4(e4)
        e5 = self.MaxPool(e4); e5 = self.Conv5(e5)

        # Decoder
        d5 = self.Up5(e5); s4 = self.Att5(gate=d5, skip_connection=e4); d5 = torch.cat((s4, d5), dim=1); d5 = self.UpConv5(d5)
        d4 = self.Up4(d5); s3 = self.Att4(gate=d4, skip_connection=e3); d4 = torch.cat((s3, d4), dim=1); d4 = self.UpConv4(d4)
        d3 = self.Up3(d4); s2 = self.Att3(gate=d3, skip_connection=e2); d3 = torch.cat((s2, d3), dim=1); d3 = self.UpConv3(d3)
        d2 = self.Up2(d3); d2 = torch.cat((e1, d2), dim=1); d2 = self.UpConv2(d2)

        # Regression output
        out = self.last(d2)
        out = F.adaptive_avg_pool2d(out, (16, 16))  # Pool to a fixed size
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    


##### Resnext50

class ResNeXt50(nn.Module):
    def __init__(self, in_channels, out_features):
        super(ResNeXt50, self).__init__()
        # Initialize ResNeXt50
        self.resnext = models.resnext50_32x4d(pretrained=True)
        # we modify the first convolution layer to accept 1 channel input
        self.resnext.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # we modify the output layer to output the desired number of values (e.g., 6 for 3 landmarks with (x, y) coordinates)
        num_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Linear(num_features, out_features)

    def forward(self, x):
        # Forward pass through the modified ResNeXt50 model
        x = self.resnext(x)
        return x
    