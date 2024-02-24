import torch
import torch.nn as nn

class ConvDropoutNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=(1,1))
        self.norm = nn.InstanceNorm2d(out_channels, eps=1e-5, momentum=stride, affine=True, track_running_stats=False)
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.all_modules = nn.Sequential(self.conv, self.norm, self.nonlin)

    def forward(self, x):
        return self.all_modules(x)


class StackedConvBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.convs = nn.Sequential(
            ConvDropoutNormRelu(in_channels, out_channels, stride),
            ConvDropoutNormRelu(out_channels, out_channels, 1)
        )

    def forward(self, x):
        return self.convs(x)


class PlainConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.Sequential(
            nn.Sequential(
                StackedConvBlocks(1, 32, 1)
            ),
            nn.Sequential(
                StackedConvBlocks(32, 64, 2),
            ),
            nn.Sequential(
                StackedConvBlocks(64, 128, 2),
            ),
            nn.Sequential(
                StackedConvBlocks(128, 256, 2),
            ),
            nn.Sequential(
                StackedConvBlocks(256, 512, 2),
            ),
            nn.Sequential(
                StackedConvBlocks(512, 512, 2),
            ),
            nn.Sequential(
                StackedConvBlocks(512, 512, 2),
            ),
        )

    def forward(self, x):
        enc_features = []
        for stage in self.stages:
            x = stage(x)
            enc_features.append(x)
        return enc_features


class UNetDecoder(nn.Module):
    def __init__(self, return_features=False):
        super().__init__()
        self.return_features = return_features
        self.encoder = PlainConvEncoder()
        self.stages = nn.ModuleList([
            StackedConvBlocks(1024,512,1),
            StackedConvBlocks(1024,512,1),
            StackedConvBlocks(512,256,1),
            StackedConvBlocks(256,128,1),
            StackedConvBlocks(128,64,1),
            StackedConvBlocks(64,32,1),
        ])
        self.transpconvs = nn.ModuleList([
            nn.ConvTranspose2d(512, 512, 2, 2),
            nn.ConvTranspose2d(512, 512, 2, 2),
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ConvTranspose2d(64, 32, 2, 2),
        ])
        self.seg_layers = nn.ModuleList([
            nn.Conv2d(512, 2, 1),
            nn.Conv2d(512, 2, 1),
            nn.Conv2d(256, 2, 1),
            nn.Conv2d(128, 2, 1),
            nn.Conv2d(64, 2, 1),
            nn.Conv2d(32, 2, 1),
        ])

    def forward(self, enc_features):
        # Initializing the decoder input with the encoder's last feature map
        dec_input = enc_features[-1]
        # print(dec_input.shape)
        
        dec_features = []
        # Decoding
        for i in range(len(self.stages)):
            # print("i: ", i)
            dec_input = self.transpconvs[i](dec_input)
            # print(dec_input.shape, enc_features[-i - 2].shape)
            dec_input = torch.cat([dec_input, enc_features[-i - 2]], dim=1)
            # print(dec_input.shape)
            dec_input = self.stages[i](dec_input)
            # print(dec_input.shape)
            dec_features.append(dec_input)
            # print("------------------------------")

        # Final segmentation
        if self.return_features == True:
            seg_outputs = []
            for i in range(len(self.seg_layers)):
                seg_outputs.append(self.seg_layers[i](dec_features[i]))
        else:
            seg_outputs = self.seg_layers[-1](dec_features[-1])

        return seg_outputs



class NNUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PlainConvEncoder()
        self.decoder = UNetDecoder()

    def forward(self, x):
        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features)
        return dec_features
