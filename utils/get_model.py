import sys
from torch import nn
import torchvision
from torchvision.models import mobilenet_v2

class MobileNetTwoHeads(nn.Module):
    def __init__(self, num_classes, num_classes2=None, dropout_p=0.2, pretrained=False):
        super(MobileNetTwoHeads, self).__init__()
        self.model = mobilenet_v2(pretrained=pretrained)
        if num_classes2 is None:
            num_classes2 = num_classes
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Identity() # needed for the forward pass to return features internally
        self.classifier1 = nn.Sequential(nn.Dropout(p=dropout_p, inplace=False),
                                         nn.Linear(num_ftrs, num_classes))
        self.classifier2 = nn.Sequential(nn.Dropout(p=dropout_p, inplace=False),
                                         nn.Linear(num_ftrs, num_classes2))

    def forward(self, x):
        x = self.model(x)
        out1 = self.classifier1(x)
        out2 = self.classifier2(x)
        return out1, out2


def get_arch(model, num_classes, additional_classes=1, dropout_p=0.0, pretrained=True):

    if model == 'mobilenet':
        model = mobilenet_v2(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=dropout_p, inplace=False),
                                         nn.Linear(num_ftrs, num_classes))
    elif model == 'mobilenet_2heads':
        model = MobileNetTwoHeads(num_classes, additional_classes, dropout_p=dropout_p, pretrained=pretrained)
    else:
        sys.exit('not a valid model_name, check models.get_model.py')

    return model

# to modify models for adding auxiliary branches I follow cadene's approach
# see, for example, https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

