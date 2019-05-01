import torch
import torch.nn as nn
from torchvision import models,transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import get_model_and_loss

def display_img(img):
    # img is a tensor
    fake_img = img.clone()
    unloader = transforms.ToPILImage()
    fake_img = unloader(fake_img)
    plt.imshow(fake_img)
    plt.show()


def read_image(style_img_path, content_img_path,img_size = 128):
    loader = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    
    style_img = Image.open(style_img_path)
    content_img = Image.open(content_img_path)

    style_img = loader(style_img)
    content_img = loader(content_img)

    return style_img, content_img
def get_optimizer(input_img):
    opt = torch.optim.LBFGS([input_img.requires_grad_()])
    return opt

def train_model(style_img, content_img,input_img,style_weight=1000000,content_weight = 1):
    model, style_losses, content_losses = get_model_and_loss(style_img,content_img)
    opt = get_optimizer(input_img)
    epoch = 30
    print('Optiming...')
    for e in range(epoch):
        print('epoch: {}...'.format(e))
        def closure():
            input_img.data.clamp_(0,1)
            opt.zero_grad()

            model.forward(input_img)

            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score+=sl.loss
            for sl in content_losses:
                content_score+=sl.loss
            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            return style_score + content_score

        opt.step(closure)

    input_img.data.clamp_(0,1)
    return input_img

def main():
    img_size = 128
    style_img_path = 'picasso.jpg'
    content_img_path = 'dancing.jpg'

    style_img, content_img = read_image(style_img_path,content_img_path)

    style_img = style_img.view((-1,3,img_size,img_size))
    content_img = content_img.view((-1,3,img_size,img_size))
    input_img = content_img.clone()

    output_img = train_model(style_img,content_img,input_img).view(3,img_size,img_size)

    display_img(output_img)
    pass
main()