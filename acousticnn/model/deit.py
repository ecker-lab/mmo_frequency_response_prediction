from torch import nn
from transformers import ViTModel
from torchvision import transforms
import torch
from transformers import ViTConfig
from transformers import SwinConfig, SwinModel


def fourier_transform(image_batch):
    # Performing Fourier Transform on the batch
    ft_image_batch = torch.fft.fftn(image_batch, dim=[2, 3])    
    # Taking absolute value to get magnitude spectrum and adding small value to avoid log(0)
    ft_image_batch = torch.abs(ft_image_batch)
    return ft_image_batch


class CustomViT(nn.Module):
    def __init__(self, config, pool):
        super(CustomViT, self).__init__()
        self.pool = pool
        self.resizer = transforms.Resize((128, 128))

        self.vit = ViTModel(config)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.resizer(x)
        x = self.vit(x).last_hidden_state
        if self.pool:
            x = self.avgpool(x.permute(0, 2, 1))
        return x


def get_vit(hidden_dim_size, pool=True):
    config = ViTConfig()
    config.image_size = 128
    config.layer_norm_eps = 1e-12
    config.model_type = "vit"
    config.num_attention_heads = 3
    config.num_hidden_layers = 12
    config.patch_size = 16
    config.qkv_bias = True
    config.hidden_act = "gelu"
    config.hidden_dropout_prob = 0.0
    config.intermediate_size = 768
    config.hidden_size = 192
    config.num_channels = 1
    model = CustomViT(config, pool)
    # config = SwinConfig()
    # config.image_size = 128
    # config.num_channels = 1
    # # Initializing a model (with random weights) from the microsoft/swin-tiny-patch4-window7-224 style configuration
    # model = SwinModel(config)
    return model
