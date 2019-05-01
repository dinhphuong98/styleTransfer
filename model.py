from torchvision import models
import torch
import copy
import torch.nn as nn

class ContentLoss(torch.nn.Module):
    def __init__(self,target):
        super(ContentLoss,self).__init__()
        self.target = target.detach()
    
    def forward(self,input):
        self.loss = torch.nn.functional.mse_loss(input,self.target)
        return input

def Gram_Matrix(input):
    b,d,h,w = input.size()
    # b batch, d depth, h height, w weight
    feature = input.view((b*d,h*w))
    G = torch.mm(feature,feature.t())
    return G.div(b*d*h*w)

class StyleLoss(torch.nn.Module):
    def __init__(self,target):
        super(StyleLoss,self).__init__()
        self.target = target.detach()

    def forward(self,input):
        G_target = Gram_Matrix(self.target)
        G_input = Gram_Matrix(input)
        self.loss = torch.nn.functional.mse_loss(G_input,G_target)
        return input


cnn = models.vgg19(pretrained=True).features.eval()


style_loss_layers = ['conv_4']
content_loss_layer = ['conv_1','conv_3','conv_4','conv_5']
def get_model_and_loss(style_img,content_img):

    cnn_vgg = copy.deepcopy(cnn)

    model = torch.nn.Sequential()

    style_losses = []
    content_losses = []
    i = 1
    for layer in cnn_vgg.children():
        if isinstance(layer,nn.Conv2d):
            name = 'conv_{}'.format(i)
            i+=1
        elif isinstance(layer,nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer,nn.ReLU):
            name = 'pool_{}'.format(i)
            layer = nn.ReLU(inplace = False)
        elif isinstance(layer,nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        
        model.add_module(name,layer)

        if name in style_loss_layers:
            target = model(style_img).detach()
            styleloss = StyleLoss(target)
            model.add_module('styleloss_{}'.format(i),styleloss)
            style_losses.append(styleloss)
        if name in content_loss_layer:
            feature = model(content_img).detach()
            contentloss = ContentLoss(feature)
            model.add_module('contentloss_{}'.format(i),contentloss)
            content_losses.append(contentloss)
        
    return model, style_losses, content_losses
