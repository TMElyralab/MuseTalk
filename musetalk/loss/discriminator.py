from torch import nn
import torch.nn.functional as F
import torch
from musetalk.loss.vgg_face import ImagePyramide

class DownBlock2d(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)

        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

        if norm:
            self.norm = nn.InstanceNorm2d(out_features, affine=True)
        else:
            self.norm = None
        self.pool = pool

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        if self.pool:
            out = F.avg_pool2d(out, (2, 2))
        return out


class Discriminator(nn.Module):
    """
    Discriminator similar to Pix2Pix
    """

    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512,
                 sn=False, **kwargs):
        super(Discriminator, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(num_channels if i == 0 else min(max_features, block_expansion * (2 ** i)),
                            min(max_features, block_expansion * (2 ** (i + 1))),
                            norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1), sn=sn))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        feature_maps = []
        out = x

        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        prediction_map = self.conv(out)

        return feature_maps, prediction_map


class MultiScaleDiscriminator(nn.Module):
    """ 
    Multi-scale (scale) discriminator
    """

    def __init__(self, scales=(), **kwargs):
        super(MultiScaleDiscriminator, self).__init__()
        self.scales = scales
        discs = {}
        for scale in scales:
            discs[str(scale).replace('.', '-')] = Discriminator(**kwargs)
        self.discs = nn.ModuleDict(discs)

    def forward(self, x):
        out_dict = {}
        for scale, disc in self.discs.items():
            scale = str(scale).replace('-', '.')
            key = 'prediction_' + scale
            #print(key)
            #print(x)
            feature_maps, prediction_map = disc(x[key])
            out_dict['feature_maps_' + scale] = feature_maps
            out_dict['prediction_map_' + scale] = prediction_map
        return out_dict



class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, discriminator):
        super(DiscriminatorFullModel, self).__init__()
        self.discriminator = discriminator
        self.scales = self.discriminator.scales
        print("scales",self.scales)
        self.pyramid = ImagePyramide(self.scales, 3)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.zero_tensor = None

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = torch.FloatTensor(1).fill_(0).cuda()
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def forward(self, x, generated, gan_mode='ls'):
        pyramide_real = self.pyramid(x)
        pyramide_generated = self.pyramid(generated.detach())

        discriminator_maps_generated = self.discriminator(pyramide_generated)
        discriminator_maps_real = self.discriminator(pyramide_real)

        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            if gan_mode == 'hinge':
                value = -torch.mean(torch.min(discriminator_maps_real[key]-1, self.get_zero_tensor(discriminator_maps_real[key]))) - torch.mean(torch.min(-discriminator_maps_generated[key]-1, self.get_zero_tensor(discriminator_maps_generated[key])))
            elif gan_mode == 'ls':
                value = ((1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2).mean()
            else:
                raise ValueError('Unexpected gan_mode {}'.format(self.train_params['gan_mode']))

            value_total += value

        return value_total
    
def main():
    discriminator = MultiScaleDiscriminator(scales=[1],
                                        block_expansion=32,
                                        max_features=512,
                                        num_blocks=4,
                                        sn=True,
                                        image_channel=3,
                                        estimate_jacobian=False)