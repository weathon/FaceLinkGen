import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    A convolutional block consisting of two Conv2D layers each followed by a ReLU activation.
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        conv_relu = []
        conv_relu.append(nn.Conv2d(in_channels=in_channels, out_channels=middle_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        conv_relu.append(nn.Conv2d(in_channels=middle_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        self.conv_ReLU = nn.Sequential(*conv_relu)

    def forward(self, x):
        return self.conv_ReLU(x)


class UNet(nn.Module):
    """
    A U-Net model for image-to-image tasks such as segmentation or image reconstruction.
    Consists of an encoder-decoder structure with skip connections.
    """
    def __init__(self, in_channels=81, out_channels=3):
        super().__init__()

        # Encoder (contracting path)
        self.left_conv_1 = ConvBlock(in_channels=in_channels, middle_channels=64, out_channels=64)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_2 = ConvBlock(in_channels=64, middle_channels=128, out_channels=128)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_3 = ConvBlock(in_channels=128, middle_channels=256, out_channels=256)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_4 = ConvBlock(in_channels=256, middle_channels=512, out_channels=512)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_5 = ConvBlock(in_channels=512, middle_channels=1024, out_channels=1024)

        # Decoder (expanding path)
        self.deconv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2,
                                           padding=1, output_padding=1)
        self.right_conv_1 = ConvBlock(in_channels=1024, middle_channels=512, out_channels=512)

        self.deconv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1,
                                           stride=2, output_padding=1)
        self.right_conv_2 = ConvBlock(in_channels=512, middle_channels=256, out_channels=256)

        self.deconv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1,
                                           stride=2, output_padding=1)
        self.right_conv_3 = ConvBlock(in_channels=256, middle_channels=128, out_channels=128)

        self.deconv_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2,
                                           output_padding=1, padding=1)
        self.right_conv_4 = ConvBlock(in_channels=128, middle_channels=64, out_channels=64)

        # Final output convolution (1x1) to reduce to desired number of output channels
        self.right_conv_5 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def encode(self, x):
        """
        Encoder path: applies a sequence of convolutional blocks and downsampling.
        Returns intermediate features for skip connections.
        """
        feature_1 = self.left_conv_1(x)
        print(f"feature_1: {feature_1.shape}")
        feature_1_pool = self.pool_1(feature_1)
    
        feature_2 = self.left_conv_2(feature_1_pool)
        print(f"feature_2: {feature_2.shape}")
        feature_2_pool = self.pool_2(feature_2)
    
        feature_3 = self.left_conv_3(feature_2_pool)
        print(f"feature_3: {feature_3.shape}")
        feature_3_pool = self.pool_3(feature_3)
    
        feature_4 = self.left_conv_4(feature_3_pool)
        print(f"feature_4: {feature_4.shape}")
        feature_4_pool = self.pool_4(feature_4)
    
        feature_5 = self.left_conv_5(feature_4_pool)
        print(f"feature_5: {feature_5.shape}")
    
        return feature_1, feature_2, feature_3, feature_4, feature_5
    
    def decode(self, feature_1, feature_2, feature_3, feature_4, feature_5, extract_features=False):
        """
        Decoder path: applies upsampling and convolution, using skip connections.
        Returns the output image and optionally intermediate features.
        """
        de_feature_1 = self.deconv_1(feature_5)
        print(f"de_feature_1: {de_feature_1.shape}")
        print(f"cat_1: {[feature_4.shape, de_feature_1.shape]}")
        temp = torch.cat((feature_4, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp)
    
        de_feature_2 = self.deconv_2(de_feature_1_conv)
        print(f"de_feature_2: {de_feature_2.shape}")
        print(f"cat_2: {[feature_3.shape, de_feature_2.shape]}")
        temp = torch.cat((feature_3, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp)
    
        de_feature_3 = self.deconv_3(de_feature_2_conv)
        print(f"de_feature_3: {de_feature_3.shape}")
        print(f"cat_3: {[feature_2.shape, de_feature_3.shape]}")
        temp = torch.cat((feature_2, de_feature_3), dim=1)
        de_feature_3_conv = self.right_conv_3(temp)
    
        de_feature_4 = self.deconv_4(de_feature_3_conv)
        print(f"de_feature_4: {de_feature_4.shape}")
        print(f"cat_4: {[feature_1.shape, de_feature_4.shape]}")
        temp = torch.cat((feature_1, de_feature_4), dim=1)
        de_feature_4_conv = self.right_conv_4(temp)
    
        out = self.right_conv_5(de_feature_4_conv)
        print(f"final_output: {out.shape}")
    
        if extract_features:
            return out, (de_feature_1_conv, de_feature_2_conv, de_feature_3_conv, de_feature_4_conv)
        else:
            return out

    def forward(self, x):
        """
        Full forward pass of the U-Net model: encoding followed by decoding.
        """
        feature_1, feature_2, feature_3, feature_4, feature_5 = self.encode(x)
        out = self.decode(feature_1, feature_2, feature_3, feature_4, feature_5)
        return out
