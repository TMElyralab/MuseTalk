import torch
import torch.nn as nn
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from musetalk.loss.discriminator import MultiScaleDiscriminator,DiscriminatorFullModel
import musetalk.loss.vgg_face as vgg_face

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

def set_requires_grad(net, requires_grad=False):
    if net is not None:
        for param in net.parameters():
            param.requires_grad = requires_grad

if __name__ == "__main__":
    cfg = OmegaConf.load("config/audio_adapter/E7.yaml")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pyramid_scale = [1, 0.5, 0.25, 0.125]
    vgg_IN = vgg_face.Vgg19().to(device)
    pyramid = vgg_face.ImagePyramide(cfg.loss_params.pyramid_scale, 3).to(device)
    vgg_IN.eval()
    downsampler = Interpolate(size=(224, 224), mode='bilinear', align_corners=False)
    
    image = torch.rand(8, 3, 256, 256).to(device)
    image_pred = torch.rand(8, 3, 256, 256).to(device)
    pyramide_real = pyramid(downsampler(image))
    pyramide_generated = pyramid(downsampler(image_pred))
    

    loss_IN = 0
    for scale in cfg.loss_params.pyramid_scale:
        x_vgg = vgg_IN(pyramide_generated['prediction_' + str(scale)])
        y_vgg = vgg_IN(pyramide_real['prediction_' + str(scale)])
        for i, weight in enumerate(cfg.loss_params.vgg_layer_weight):
            value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean() 
            loss_IN += weight * value
    loss_IN /= sum(cfg.loss_params.vgg_layer_weight)  # 对vgg不同层取均值，金字塔loss是每层叠
    print(loss_IN)

    #print(cfg.model_params.discriminator_params)

    discriminator = MultiScaleDiscriminator(**cfg.model_params.discriminator_params).to(device)
    discriminator_full = DiscriminatorFullModel(discriminator)
    disc_scales = cfg.model_params.discriminator_params.scales
    # Prepare optimizer and loss function
    optimizer_D = optim.AdamW(discriminator.parameters(), 
                                lr=cfg.discriminator_train_params.lr, 
                                weight_decay=cfg.discriminator_train_params.weight_decay,
                                betas=cfg.discriminator_train_params.betas,
                                eps=cfg.discriminator_train_params.eps)
    scheduler_D = CosineAnnealingLR(optimizer_D, 
                                    T_max=cfg.discriminator_train_params.epochs, 
                                    eta_min=1e-6)

    discriminator.train()

    set_requires_grad(discriminator, False)

    loss_G = 0.
    discriminator_maps_generated = discriminator(pyramide_generated)
    discriminator_maps_real = discriminator(pyramide_real)

    for scale in disc_scales:
        key = 'prediction_map_%s' % scale
        value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
        loss_G += value

    print(loss_G)
