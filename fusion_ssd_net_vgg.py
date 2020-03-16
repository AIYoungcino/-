import torch
import torch.nn as nn
import l2norm
class UpsampleLayer(torch.nn.Module):

    def __init__(self):
        super(UpsampleLayer, self).__init__()

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')


class ConvolutionalLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.sub_module(x)

class SSD(torch.nn.Module):

    def __init__(self):
        super(SSD, self).__init__()
        self.trunk_38 = torch.nn.Sequential(
            ConvolutionalLayer(3, 64, 3, 1, 1),#300 11  1080
            ConvolutionalLayer(64, 64, 3, 1, 1),#12
            ConvolutionalLayer(64, 64, 2, 2, 0),#150 MAX1   640
            ConvolutionalLayer(64, 128, 3, 1, 1),#21
            ConvolutionalLayer(128, 128, 3, 1, 1),#22
            #ConvolutionalLayer(128, 128, 3, 1, 1),
            ConvolutionalLayer(128, 128, 2, 2, 0),#75 MAX2   320
            ConvolutionalLayer(128, 256, 3, 1, 1),#31
            ConvolutionalLayer(256, 256, 3, 1, 1),#32
            ConvolutionalLayer(256, 256, 3, 1, 1),#33
            ConvolutionalLayer(256, 256, 2, 2, 0),#MAX3 38   160
            ConvolutionalLayer(256, 512, 3, 1, 1),  # 41
            ConvolutionalLayer(512, 512, 3, 1, 1),  # 42
            ConvolutionalLayer(512, 512, 3, 1, 1),#CONV43小目标检测

            #ConvolutionalLayer(512, 1024, 3, 1, 1),
            #UpsampleLayer(512, 1024)
        )

        self.up_38 = torch.nn.Sequential(
            ConvolutionalLayer(512, 512, 1, 1, 0),#38 160
            UpsampleLayer()
        )
        #self.convset_38 = torch.nn.Sequential(
        #    ConvolutionalLayer(512, 512, 3, 1, 1)
        #)
        self.trunk_19 = torch.nn.Sequential(
            #ConvolutionalLayer(512, 512, 3, 1, 1),
            ConvolutionalLayer(512, 512, 2, 2, 0),#pool4, 80
            ConvolutionalLayer(512, 512, 3, 1, 1),#19 51  80
            ConvolutionalLayer(512, 512, 3, 1, 1),#52
            ConvolutionalLayer(512, 512, 3, 1, 1),#53
            ConvolutionalLayer(512, 512, 3, 1, 1),#pool5
            #ConvolutionalLayer(512, 512, 1, 2, 0),
        )
        self.trunk_19_1 = torch.nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),#300 conv6
            ConvolutionalLayer(1024, 1024, 3, 1, 1),#conv7 10
            #UpsampleLayer(1024, 512)
        )
        self.trunk_10 = torch.nn.Sequential(
            ConvolutionalLayer(1024, 256, 3, 1, 1),#19
            ConvolutionalLayer(256, 512, 2, 2, 0),#10  40
        )
        self.trunk_5 = torch.nn.Sequential(
            ConvolutionalLayer(512, 128, 1, 1, 1),
            ConvolutionalLayer(128, 256, 4, 2, 1),#5 20
        )
        self.trunk_3 = torch.nn.Sequential(
            ConvolutionalLayer(256, 128, 1, 1, 1),
            ConvolutionalLayer(128, 256, 3, 2, 1),
        )
        self.trunk_1 = torch.nn.Sequential(
            ConvolutionalLayer(256, 128, 1, 1, 1),
            ConvolutionalLayer(128, 256, 3, 1, 1),
        )



        # 特征层位置输出
        self.feature_map_loc_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_loc_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_loc_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_loc_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_loc_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_loc_6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)
        )
        # 特征层类别输出
        self.feature_map_conf_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=4 * 21, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_conf_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=6 * 21, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_conf_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=6 * 21, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_conf_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=6 * 21, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_conf_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=4 * 21, kernel_size=3, stride=1, padding=1)
        )
        self.feature_map_conf_6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=4 * 21, kernel_size=3, stride=1, padding=1)
        )

    # 正向传播过程
    def forward(self, image):
        my_L2Norm = l2norm.L2Norm(1024, 20)
       #h_38 = self.trunk_19(image)

        feature_map_1 = self.trunk_38(image)#512

        #feature_map_2 = self.trunk_19(feature_map_1)
        h_38_out = self.trunk_19(feature_map_1)
        h_38 = self.up_38(h_38_out)
        #convset_out_38 = self.convset_38(feature_map_1)
        #up_out_38 = self.up_38(convset_out_38)
        feature_map_1 = torch.cat((h_38, feature_map_1), dim=2)  # out
        feature_map_1 = my_L2Norm(feature_map_1)
        loc_1 = self.feature_map_loc_1(feature_map_1).permute((0, 2, 3, 1)).contiguous()
        conf_1 = self.feature_map_conf_1(feature_map_1).permute((0, 2, 3, 1)).contiguous()

        feature_map_2 = self.trunk_19_1(h_38_out)
        loc_2 = self.feature_map_loc_2(feature_map_2).permute((0, 2, 3, 1)).contiguous()
        conf_2 = self.feature_map_conf_2(feature_map_2).permute((0, 2, 3, 1)).contiguous()
        #out = self.trunk_19(image)
        #out = self.conv8_2(out)

        feature_map_3 = self.trunk_10(feature_map_2)
        loc_3 = self.feature_map_loc_3(feature_map_3).permute((0, 2, 3, 1)).contiguous()
        conf_3 = self.feature_map_conf_3(feature_map_3).permute((0, 2, 3, 1)).contiguous()
        #out = self.conv9_1(self.trunk_10)
        #out = self.conv9_2(out)
        feature_map_4 = self.trunk_5(feature_map_3)
        loc_4 = self.feature_map_loc_4(feature_map_4).permute((0, 2, 3, 1)).contiguous()
        conf_4 = self.feature_map_conf_4(feature_map_4).permute((0, 2, 3, 1)).contiguous()
        #out = self.conv10_1(self.trunk_5)
        #out = self.conv10_2(out)
        feature_map_5 = self.trunk_3(feature_map_4)
        loc_5 = self.feature_map_loc_5(feature_map_5).permute((0, 2, 3, 1)).contiguous()
        conf_5 = self.feature_map_conf_5(feature_map_5).permute((0, 2, 3, 1)).contiguous()
        #out = self.conv11_1(self.trunk_3)
        #out = self.conv11_2(out)
        feature_map_6 = self.trunk_1(feature_map_5)
        loc_6 = self.feature_map_loc_6(feature_map_6).permute((0, 2, 3, 1)).contiguous()
        conf_6 = self.feature_map_conf_6(feature_map_6).permute((0, 2, 3, 1)).contiguous()
        loc_list = [loc_1, loc_2, loc_3, loc_4, loc_5, loc_6]
        conf_list = [conf_1, conf_2, conf_3, conf_4, conf_5, conf_6]
        return loc_list, conf_list