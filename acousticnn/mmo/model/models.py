import math
import torch
from torch import nn
import numpy as np
import math
from .encoding import *
from .transformer import TransformerEncoder
from torchvision.ops import MLP


def add_new_dim(tensor):
    return tensor.unsqueeze(-1)

class ImplicitTransformer(nn.Module):
    def __init__(self, parameters_example, config):
        super().__init__()
        num_encoders = len(parameters_example)
        num_pars = torch.hstack(parameters_example).shape[1]
        self.config = config 
        self.input_encoding = get_input_encoding(config.input_encoding, config.encoding_dim)
        if config.input_encoding == "none":
            config.encoding_dim = 1
            
        self.pos_encoding = PositionalEncoding(config.encoder.embed_dim, base=10000)

        self.groupwise_projection = GroupwiseProjection(num_encoders, int(config.encoding_dim), config.encoder.embed_dim)
        self.frequency_projection = nn.Linear(config.encoding_dim, config.decoder.embed_dim) 
        self.embedding_projection = nn.Linear(config.encoder.embed_dim, config.decoder.embed_dim) 
        self.transformer = TransformerEncoder(**config.encoder)
        self.decoder = TransformerEncoder(**config.decoder)
        self.linear_projection2 = nn.Linear(config.decoder.embed_dim, 1)
        self.apply(weights_init)

    def forward(self, frequency, parameters):
        self.num_masses = parameters[0].shape[1]
        len_groups = [group.shape[1] for group in parameters]
        B, num_frequencies = frequency.shape
        embeddings = self.input_encoding(torch.hstack(parameters)) 
        pos_encoding = self.pos_encoding(torch.cat(parameters, dim=1))
        embeddings = self.groupwise_projection(embeddings, len_groups)

        # encoder
        amplitude_embedding = self.transformer(embeddings, pos_encoding) + embeddings # B x num_tokens x token_dim
        amplitude_embedding = amplitude_embedding[:, :self.num_masses] # B x num_mass tokens x token_dim
        amplitude_embedding = amplitude_embedding.repeat_interleave(num_frequencies, dim=0)

        amplitude_embedding = self.embedding_projection(amplitude_embedding) # B x num_mass tokens x token_dim

        # decoder
        frequency_embeddings = self.input_encoding(frequency) 
        frequency_embeddings = self.frequency_projection(frequency_embeddings) # B x num_frequencies x token_dim
        frequency_embeddings = frequency_embeddings.view(B*num_frequencies, 1, self.config.decoder.embed_dim) # transmit num_frequencies into Batch to enable parallel processing
        frequency_embeddings = frequency_embeddings.repeat(1, self.num_masses, 1) # In order to fit to the number of mass tokens repeat along token axis
        amplitudes = frequency_embeddings + self.decoder(amplitude_embedding, frequency_embeddings) #B * num_frequencies x num_masses x token_dim
        amplitudes = self.linear_projection2(amplitudes)

        return amplitudes.view(B, num_frequencies, self.num_masses)


class ImplicitMLP(nn.Module):
    def __init__(self, parameters_example, config):
        super().__init__()
        num_encoders = len(parameters_example) + 1
        num_parameters = torch.hstack(parameters_example).shape[1] + 1
        self.num_masses = parameters_example[0].shape[1]

        in_channels = config.embed_dim*num_parameters
        self.input_encoding = get_input_encoding(config.input_encoding, config.encoding_dim)
        if config.input_encoding == "none":
            config.encoding_dim = 1

        self.groupwise_projection = GroupwiseProjection(num_encoders, int(config.encoding_dim), config.embed_dim)

        self.internal_net = MLP(in_channels=in_channels, hidden_channels=config.depth*[config.mlp_width])
        self.output_layer = nn.Linear(in_features=config.mlp_width, out_features=self.num_masses)
        self.apply(weights_init)

    def forward(self, frequency, parameters):
        B, num_frequencies = frequency.shape # B x 200
        len_groups = [group.shape[1] for group in parameters] + [1]
        parameters = torch.hstack(parameters) # b x num_parameters
        parameters = parameters.repeat_interleave(num_frequencies, dim=0) # B*200 x num_parameters
        parameters = torch.hstack((parameters, frequency.view(-1, 1)))

        encoding = self.input_encoding(parameters)
        encoding = self.groupwise_projection(encoding, len_groups)
        prediction = self.internal_net(encoding.view(B*num_frequencies, -1))
        prediction = self.output_layer(prediction)

        return prediction.view(B, num_frequencies, self.num_masses)


def get_input_encoding(encoding_type: str, encoding_dim):
    if encoding_type == "none":
        return add_new_dim
    elif encoding_type == "sin":
        return SinosoidalEncoding(dim=encoding_dim)
    elif encoding_type == "random":
        return RandomEncoding(dim=encoding_dim, factor_pars=(0, 1))
    else:
        raise NonImplementedError


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        #if not m.in_features == m.out_features:
        torch.nn.init.xavier_uniform_(m.weight)
        #else:
        #    torch.nn.init.eye_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.normal_(m.weight, mean=1, std=0.02)
        torch.nn.init.zeros_(m.bias)


class GroupwiseProjection(nn.Module):
    def __init__(self, num_groups, input_dim, output_dim):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_groups)])

    def forward(self, x, len_groups=None):
        """This class applies separate nn.Linear layers with separate weights on an input tensor and returns the result

        Args:
            x (Tensor): Size: Batch x Tokens x token_dim
            len_groups (list): list in the token dimension, that informs which tokens are groupped together

        Returns:
            tensor: _description_
        """
        assert len(len_groups) == len(self.projections)
        assert np.sum(len_groups) == x.shape[1]
        projected_list = []
        current = len_groups[0]
        x_group = x[:, 0:current]
        for i, projection in enumerate(self.projections):
            projected = projection(x_group)
            projected_list.append(projected)
            last = current
            if not i == len(self.projections) -1:
                current += len_groups[i+1]
                x_group = x[:, last:current]
        return torch.cat(projected_list, dim=1)

