import torch
import torch.nn as nn
import t3nsor as t3
#from .utils import load_state_dict_from_url

#when using from ptt_vgg import * outside, only all variables are imported
__all__ = [
    'ptt_VGG', 'ptt_vgg11', 'ptt_vgg11_bn', 'ptt_vgg13', 'ptt_vgg13_bn', 'ptt_vgg16', 'ptt_vgg16_bn',
    'ptt_vgg19_bn', 'ptt_vgg19',
]


# model_urls = {
#     'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
#     'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
#     'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
#     'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#     'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
#     'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
#     'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
#     'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
# }

class tt_conv_VGG(nn.Module):

    def __init__(self, features, type, num_classes=1000):
        super(tt_conv_VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        if type == 'tt':
            self.classifier = nn.Sequential(
                t3.TTLinear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                t3.TTLinear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                t3.TTLinear(4096, num_classes),
        )
        elif type == 'ptt_solver':
            self.classifier = nn.Sequential(
                t3.Solver(512 * 7 * 7, 4096),
                nn.Dropout(),
                t3.Solver(4096, 4096),
                nn.Dropout(),
                t3.Solver(4096, num_classes),
            )
        elif type == 'test':
            self.classifier = nn.Linear(512*7*7,num_classes)
        else:
            print('wrong name!')
        # if init_weights:
        #     self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = x.full()
        x = self.avgpool(x)
        #flatten begin from the dim next of batch size
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)


class ptt_VGG(nn.Module):

    def __init__(self, features, type, num_classes=1000, init_weights=False):
        super(ptt_VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        if type == 'ptt':
            self.classifier = nn.Sequential(
                t3.TTLinear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                t3.TTLinear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                t3.TTLinear(4096, num_classes),
        )
        elif type == 'ptt_solver':
            self.classifier = nn.Sequential(
                t3.TTSolver(512 * 7 * 7, 4096),
                nn.Dropout(),
                t3.TTSolver(4096, 4096),
                nn.Dropout(),
                t3.TTSolver(4096, num_classes),
            )
        else:
            print('wrong name!')
        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        #flatten begin from the dim next of batch size
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_ttconv_layers(cfg_tt):
    layers = []
    for v in cfg:
        if v == 'M':
            shape = []
            conv2d = TTconv(shape,N=3)
            layers +=  conv2d
        else:
            #TODO: fill shape
            shape = []
            conv2d = TTconv(shape,N=1)
            layers += conv2d

    return nn.Sequential(*layers)


cfgs_tt = {
    'A': ['M', 128],
    'B': ['M', 128],
    'D': ['M', 128],
    'E': ['M', 128],
}

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _ptt_vgg(arch, cfg, batch_norm, pretrained, progress, type, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = ptt_VGG(features=make_layers(cfgs[cfg], batch_norm=batch_norm), type=type, **kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
        raise NotImplementedError
    return model

def _tt_conv_vgg(arch, cfg, batch_norm, pretrained, progress, type, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = tt_conv_VGG(features=make_layers(cfgs[cfg], batch_norm=batch_norm), type=type, **kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        # model.load_state_dict(state_dict)
        raise NotImplementedError
    return model


def ptt_vgg11(type, pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ptt_vgg('ptt_vgg11', 'A', False, pretrained, progress, type, **kwargs)


def ptt_vgg11_bn(type, pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ptt_vgg('ptt_vgg11_bn', 'A', True, pretrained, progress, type, **kwargs)


def ptt_vgg13(type, pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ptt_vgg('ptt_vgg13', 'B', False, pretrained, progress, type, **kwargs)


def ptt_vgg13_bn(type, pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ptt_vgg('ptt_vgg13_bn', 'B', True, pretrained, progress, type, **kwargs)


def ptt_vgg16(type, pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ptt_vgg('ptt_vgg16', 'D', False, pretrained, progress, type, **kwargs)


def ptt_vgg16_bn(type, pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ptt_vgg('ptt_vgg16_bn', 'D', True, pretrained, progress, type, **kwargs)


def ptt_vgg19(type, pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ptt_vgg('ptt_vgg19', 'E', False, pretrained, progress, type, **kwargs)


def ptt_vgg19_bn(type, pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ptt_vgg('ptt_vgg19_bn', 'E', True, pretrained, progress, type, **kwargs)
