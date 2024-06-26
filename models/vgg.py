import torch
from torch import nn
from typing import List, Union, cast, Dict

class VGGNet(nn.Module):
    '''
    VGGNet constructed from "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    '''
    def __init__(
        self, config: str = 'A', num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5, batch_norm: bool = False):
        super().__init__()

        cfg = self._config(config)
        features = self._make_layers(cfg, batch_norm)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _config(self, config):
        cfgs: Dict[str, List[Union[str, int]]] = {
            "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
            "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
        }
        return cfgs[config]

    def _make_layers(self, cfg: List[Union[str, int]], batch_norm: bool = False):        
        layers: List[nn.Module] = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.LeakyReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
                