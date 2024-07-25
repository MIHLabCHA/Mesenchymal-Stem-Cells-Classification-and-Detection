import torchvision.models as models
from torch import nn
from collections import OrderedDict


# Classification
class CCCL(nn.Module):
    def __init__(self, FE_arch='ResNet50', pretrained=True, channels=3, num_classes=2):
        super(CCCL, self).__init__()
        self.FE_arch = FE_arch
        if FE_arch=='ResNet50':
            self.FE = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.FE.conv1 = nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif FE_arch=='Inceptionv3':
            self.FE = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT if pretrained else None)
            self.FE.Conv2d_1a_3x3 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=(3, 3), stride=(2, 2), bias=False)
            self.bn = nn.BatchNorm2d(num_features=32)
        elif FE_arch=='AlexNet':
            self.FE = models.alexnet(weights=models.AlexNet_Weights.DEFAULT if pretrained else None)
            self.alexnet_dropout = nn.Dropout(0.5)
            self.alexnet_fc = nn.Linear(in_features=9216, out_features= 2048, bias=True)
        elif FE_arch=='ShuffleNetv2':
            self.FE = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.DEFAULT if pretrained else None)
            self.shufflenet_avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        elif FE_arch=='MobileNetv3':
            self.FE = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
            self.mobilenetv3_fc = nn.Linear(in_features=960, out_features=2048, bias=True)
        elif FE_arch=='MNASNet':
            self.FE = models.mnasnet1_3(weights=models.MNASNet1_3_Weights.DEFAULT if pretrained else None)
            self.mnasnet_avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.mnasnet_fc = nn.Linear(in_features=1280, out_features=2048, bias=True)
        elif FE_arch=='EfficientNetv2':
            self.FE = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT if pretrained else None)
            self.efficientnetv2_fc = nn.Linear(in_features=1280, out_features=2048, bias=True)
        elif FE_arch=='VitTrans':
            self.FE = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT if pretrained else None)
            self.vittrans_fc = nn.Linear(in_features=1000, out_features=2048, bias=True)
        elif FE_arch=='custom':
            self.FE = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=3, stride=1, padding=1)),
                ('dropout1', nn.Dropout2d()),
                ('bn1', nn.BatchNorm2d(num_features=128)),
                ('relu1', nn.ReLU()),
                ('pool1', nn.MaxPool2d(kernel_size=2)),
                ('conv2', nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1)),
                ('dropout2', nn.Dropout2d()),
                ('bn2', nn.BatchNorm2d(num_features=512)),
                ('relu2', nn.ReLU()),
                ('pool2', nn.MaxPool2d(kernel_size=2)),
                ('conv3', nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=3, stride=1, padding=1)),
                ('dropout3', nn.Dropout2d()),
                ('bn3', nn.BatchNorm2d(num_features=2048)),
                ('relu3', nn.ReLU()),
                ('pool3', nn.MaxPool2d(kernel_size=2)),
                ('avgpool', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
            ]))
        self.FC1 = nn.Linear(in_features=2048, out_features=1024)
        self.FC2 = nn.Linear(in_features=1024, out_features=num_classes if num_classes>2 else 1)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        if self.FE_arch=='ResNet50':
            x = self.FE.conv1(x)
            x = self.FE.bn1(x)
            x = self.FE.relu(x)
            x = self.FE.maxpool(x)
            x = self.FE.layer1(x)
            x = self.FE.layer2(x)
            x = self.FE.layer3(x)
            x = self.FE.layer4(x)
            x = self.FE.avgpool(x)
        elif self.FE_arch=='Inceptionv3':
            x = self.FE.Conv2d_1a_3x3(x)
            x = self.bn(x)
            x = self.FE.Conv2d_2a_3x3(x)
            x = self.FE.Conv2d_2b_3x3(x)
            x = self.FE.maxpool1(x)
            x = self.FE.Conv2d_3b_1x1(x)
            x = self.FE.Conv2d_4a_3x3(x)
            x = self.FE.maxpool2(x)
            x = self.FE.Mixed_5b(x)
            x = self.FE.Mixed_5c(x)
            x = self.FE.Mixed_5d(x)
            x = self.FE.Mixed_6a(x)
            x = self.FE.Mixed_6b(x)
            x = self.FE.Mixed_6c(x)
            x = self.FE.Mixed_6d(x)
            x = self.FE.Mixed_6e(x)
            x = self.FE.Mixed_7a(x)
            x = self.FE.Mixed_7b(x)
            x = self.FE.Mixed_7c(x)
            x = self.FE.avgpool(x)
        elif self.FE_arch=='AlexNet':
            x = self.FE.features(x)
            x = self.FE.avgpool(x)
            x = nn.functional.relu(self.alexnet_dropout(x.view(x.size(0), -1)))
            x = self.alexnet_fc(x)
        elif self.FE_arch=='ShuffleNetv2':
            x = self.FE.conv1(x)
            x = self.FE.maxpool(x)
            x = self.FE.stage2(x)
            x = self.FE.stage3(x)
            x = self.FE.stage4(x)
            x = self.FE.conv5(x)
            x = self.shufflenet_avgpool(x)
        elif self.FE_arch=='MobileNetv3':
            x = self.FE.features(x)
            x = self.FE.avgpool(x)
            x = self.mobilenetv3_fc(x.view(x.size(0), -1))
        elif self.FE_arch=='MNASNet':
            x = self.FE.layers(x)
            x = self.mnasnet_avgpool(x)
            x = self.mnasnet_fc(x.view(x.size(0), -1))
        elif self.FE_arch=='EfficientNetv2':
            x = self.FE.features(x)
            x = self.FE.avgpool(x)
            x = self.efficientnetv2_fc(x.view(x.size(0), -1))
        elif self.FE_arch=='VitTrans':
            x = self.FE(x)
            x = self.vittrans_fc(x.view(x.size(0), -1))
        elif self.FE_arch=='custom':
            x = self.FE(x)
        x = nn.functional.relu(self.FC1(x.view(x.size(0), -1)))
        x = self.dropout(x)
        x = self.FC2(x)
        return x
