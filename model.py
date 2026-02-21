import torch
import torch.nn as nn
import numpy as np
import math

from typing import List, Dict, Tuple, Set, Optional, Union, Any
import itertools

######################################################################################
######################################################################################

def initialize_weights(cls, init_params, widths, readout, fc=None, type_init='normal', init_scale=None, last_layer_zero_init=False, seed=None):
    """
    Initialize the weights of each trainable layer of your network using Normal or Kaming/Xavier uniform initialization.
    """
    assert type_init in ['kaiming', 'xavier', 'normal']

    if seed is not None:
        # Save the current random state
        torch_state = torch.get_rng_state()
        torch.manual_seed(seed)

    if init_params:
        with torch.no_grad():
            if len(widths)>=3:
                for layer in readout:
                    if isinstance(layer, nn.Linear):
                        if type_init=="normal":
                            torch.nn.init.normal_(layer.weight, mean=0.0, std=1.0 / math.sqrt(layer.weight.shape[0]))
                            # torch.nn.init.normal_(layer.weight, mean=0.0, std=0.25**0.5 / np.power(2*layer.weight.shape[0], 1/3))
                        elif type_init=="kaiming":
                            nn.init.kaiming_uniform_(layer.weight)
                        elif type_init=="xavier":
                            nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

            if isinstance(fc, nn.Linear):
                if type_init=="normal":
                    torch.nn.init.normal_(fc.weight, mean=0.0, std=1.0 / math.sqrt(fc.weight.shape[0]))
                    # torch.nn.init.normal_(fc.weight, mean=0.0, std=0.25**0.5 / np.power(2*fc.weight.shape[0], 1/3))
                elif type_init=="kaiming":
                    nn.init.kaiming_uniform_(fc.weight)
                elif type_init=="xavier":
                    nn.init.xavier_uniform_(fc.weight)
                if fc.bias is not None:
                    nn.init.constant_(fc.bias, 0)

    if (init_scale!=1.0) and (init_scale is not None):
        with torch.no_grad():
            for p in cls.parameters():
                p.data = init_scale * p.data

    if last_layer_zero_init and isinstance(fc, nn.Linear):
        fc.weight.data.zero_()
        if fc.bias is not None:
            fc.bias.data.zero_()

    # Restore the previous random state
    if seed is not None:
        torch.set_rng_state(torch_state)

######################################################################################
######################################################################################

def make_mlp(widths, activation_class=None, bias=True, head=[], tail=[]):
    """
    Creates a multi-layer perceptron (MLP) as a torch.nn.Sequential module based on the 
    provided widths and activation function.
    The MLP consists of linear layers defined by the widths, with optional activation functions in between

    Parameters:
        widths (List[int]): A list of integers specifying the input and output dimensions of each layer. 
                            For example, [input_dim, hidden_dim1, hidden_dim2, output_dim].
        activation_class (torch.nn.Module or None): The activation function class to use between layers (e.g., torch.nn.ReLU). 
                                                   If None, no activation functions are added.
        bias (bool): Whether to include a bias term in the linear layers.
        head (List[torch.nn.Module]): A list of torch.nn.Modules to prepend to the MLP.
        tail (List[torch.nn.Module]): A list of torch.nn.Modules to append to the MLP.
    """
    L = len(widths)
    if L <= 1 : return torch.nn.Identity()
    return torch.nn.Sequential(*(head + sum(
        [[torch.nn.Linear(i, o, bias=bias)] + ([activation_class()] if (n < L-2 and activation_class is not None) else [])
         for n, (i, o) in enumerate(zip(widths, widths[1:]))], []) + tail))

######################################################################################
######################################################################################

class Encoder_Decoder(nn.Module):
    """
    Encoder Decoder architecture.
    The model receives as input a pair of vectors (a, b) in R^{2 x p}, 
    and produces as output a distribution over the possible results of the operation.
    
    Typically, the input vectors represent the two operands of an arithmetic operation, where each operand 
    is represented as a one-hot encoded vector of size p. The input is thus a tensor of shape (batch_size, 2, p), 
    one-hot encoded, and produces as output a distribution over the possible results of the operation.

    The encoder consists of two separate MLPs (one for each input token) that produce representations 
    of the input tokens : h1 = encoder1(a), h2 = encoder2(b) in R^{d} each. 
    The two representations are then aggregated to produce a single representation h in R^{rep_dim}.
    - If aggregation_mode is 'sum', the representations of the two tokens are summed element-wise.
        h = h1 + h2, so rep_dim = d.
    - If aggregation_mode is 'concat', the representations of the two tokens are concatenated.
        h = [h1; h2], so rep_dim = 2*d.
    - If aggregation_mode is 'matrix_product', the representations of the two tokens are multiplied as matrices.
        h1 and h2 are reshaped to (sqrt(d), sqrt(d))
        h = h1 @ h2, so rep_dim = d (with the constraint that d must be a perfect square).
    - If aggregation_mode is 'hadamard_product', the representations of the two tokens are multiplied element-wise.
        h = h1 * h2, so rep_dim = d.
    """
    def __init__(
        self, 
        aggregation_mode:str, 
        widths_encoder:List, 
        widths_decoder:List, 
        activation_class_encoder=None, 
        activation_class_decoder=nn.ReLU, 
        bias_encoder:bool=True, 
        bias_decoder:bool=True, 
        bias_classifier:bool=True, 
        dropout:float=0.0, 
        init_scale:float=None, 
        init_params:bool=False, 
        type_init:str='normal', 
        seed:int=None, 
        last_layer_zero_init:bool=False
    ):
        """
        aggregation_mode (str): The method to aggregate the representations of the two tokens.
        widths_encoder (List[int]): A list of integers specifying the widths of the encoder MLPs. 
                                    The last element must be the same for both encoders and defines the representation dimension.
        widths_decoder (List[int]): A list of integers specifying the widths of the decoder MLP. 
                                    The first element must match the representation dimension defined by the encoder.
        activation_class_encoder (torch.nn.Module or None): The activation function class to use in the encoder MLPs. 
                                                            If None, no activation functions are added in the encoder.
        activation_class_decoder (torch.nn.Module or None): The activation function class to use in the decoder MLPs. 
                                                            If None, no activation functions are added in the decoder.
        bias_encoder (bool): Whether to include a bias term in the encoder linear layers.
        bias_decoder (bool): Whether to include a bias term in the decoder linear layers.
        bias_classifier (bool): Whether to include a bias term in the final classification layer.
        dropout (float): Dropout probability to apply after each layer in both encoder and decoder (except the last layer). 
                         Default is 0.0 (no dropout).
        init_scale (float or None): If not None, scales the initialized weights by this factor.
        init_params (bool): Whether to initialize the weights of the model using the specified type_init method.
        type_init (str): The type of initialization to use for the weights. Must be one of 'kaiming', 'xavier', or 'normal'.
        seed (int): Random seed for weight initialization.
        last_layer_zero_init (bool): Whether to initialize the last layer's weights to zero.
                                     When init_scale is too large, this can help stabilize training at the beginning.
        """
        super(Encoder_Decoder, self).__init__()

        assert widths_encoder[-1] == widths_decoder[0]
        assert aggregation_mode in ['sum', 'concat', 'matrix_product', 'hadamard_product']

        
        rep_dim = widths_encoder[-1]
        if aggregation_mode == 'concat':    
            widths_decoder[0] = 2 * widths_decoder[0]   
        elif aggregation_mode == 'matrix_product':  
            s = 2  
            self.rep_dim_sqrt = int(rep_dim**(1/s))
            assert self.rep_dim_sqrt**s == rep_dim, f"rep_dim ({rep_dim}) must be a perfect square for {aggregation_mode} mode."

        self.aggregation_mode = aggregation_mode
        self.activation_class_encoder = activation_class_encoder
        self.widths_encoder = widths_encoder
        self.widths_decoder = widths_decoder

        # Encoders
        self.encoder = nn.ModuleList([
            make_mlp(
                widths=widths_encoder, activation_class=activation_class_encoder, bias=bias_encoder, head=[nn.Flatten()], 
                tail=([nn.Dropout(p=dropout)] if dropout>0 else []) + ([activation_class_encoder()] if activation_class_encoder is not None else [])) 
            for _ in range(2)])

        # Decoder = readout + fc
        self.readout = make_mlp(
            widths=widths_decoder[:-1], activation_class=activation_class_decoder, bias=bias_decoder, head=[], 
            tail=([nn.Dropout(p=dropout)] if dropout>0 else []) + ([activation_class_decoder()] if activation_class_decoder is not None else []))

        self.fc = nn.Linear(widths_decoder[-2], widths_decoder[-1], bias=bias_classifier)

 
        for encoder in self.encoder :
            initialize_weights(self, init_params, widths_encoder, encoder, None, type_init, None, last_layer_zero_init, seed)
        initialize_weights(self, init_params, widths_decoder, self.readout, self.fc, type_init, init_scale, last_layer_zero_init, seed)

    def aggregate(self, h):
        """
        Aggregates the representations of the two tokens based on the specified aggregation mode.
        * If aggregation_mode is 'sum', the representations of the two tokens are summed element-wise.
        * If aggregation_mode is 'concat', the representations of the two tokens are concatenated.
        * If aggregation_mode is 'matrix_product', the representations of the two tokens are multiplied as matrices.
        * If aggregation_mode is 'hadamard_product', the representations of the two tokens are multiplied element-wise.

        Parameters:
            h (List[torch.Tensor]): A list of two tensors, each of shape (batch_size, rep_dim), representing the encoded representations of the two input tokens.

        Returns:
            torch.Tensor: The aggregated representation of the two tokens, with shape (batch_size, rep_dim*) depending on the aggregation mode.
        """
        if self.aggregation_mode == 'sum':
            h_aggreg = torch.stack(h, dim=0).sum(dim=0) # (batch_size, rep_dim*)
        elif self.aggregation_mode == 'concat':
            # Concatenate the tensors along the last dimension
            h_aggreg = torch.cat(h, dim=-1)  # (batch_size, 2 * rep_dim)
        elif self.aggregation_mode == 'matrix_product':
            # Product of all tensors 
            # Reshape each tensor to (batch_size, rep_dim_sqrt, rep_dim_sqrt)
            reshaped_h = [tensor.reshape(-1, self.rep_dim_sqrt, self.rep_dim_sqrt) for tensor in h]
            h_aggreg = torch.bmm(reshaped_h[0], reshaped_h[1]) # (batch_size, rep_dim_sqrt, rep_dim_sqrt)
            # Flatten the result back to (batch_size, rep_dim)
            h_aggreg = h_aggreg.flatten(1, -1) # (batch_size, rep_dim*)

        elif self.aggregation_mode == 'hadamard_product':
            # Hadamard (element-wise) product of all tensors 
            h_aggreg = h[0] * h[1]
        else:
            raise ValueError(f"Invalid aggregation_mode '{self.aggregation_mode}'.")

        return h_aggreg

    def tokens_to_embeddings(self, x):
        """
        Converts input tokens to their corresponding embeddings using the encoder MLPs and aggregates them.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, 2, input_dim), representing the two tokens for 
                              each example in the batch.

        Returns:
            torch.Tensor: The aggregated embeddings of the input tokens, with shape (batch_size, rep_dim*).
        """
        assert x.dim() >= 3
        assert x.shape[1] == 2
        h = [self.encoder[i](x[:,i]) for i in range(2)] # 2 * (batch_size, rep_dim*)
        return self.aggregate(h) # (batch_size, rep_dim*)

    def get_representation(self, x) :
        """
        Gets the representation of the input tokens by passing them through the encoder MLPs and the readout layer.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, 2, input_dim), representing the two tokens for 
                              each example in the batch.

        Returns:
            torch.Tensor: The representation of the input tokens, with shape (batch_size, rep_dim).
        """
        #return self.tokens_to_embeddings(x) # (batch_size, rep_dim)
        return self.readout(self.tokens_to_embeddings(x)) # (batch_size, rep_dim)

    def get_logits(self, representation) :
        """
        Gets the logits for classification by passing the representation through the final fully connected layer.

        Parameters:
            representation (torch.Tensor): The representation of the input tokens, with shape (batch_size, rep_dim).

        Returns:
            torch.Tensor: The logits for classification, with shape (batch_size, output_dim).
        """
        #return self.fc(self.readout(representation)) # (batch_size, output_dim*)
        return self.fc(representation) # (batch_size, output_dim*)

    def forward(self, x, activation=False):
        """
        Forward pass of the model. 
        Takes as input a batch of pairs of tokens and produces the corresponding logits for classification.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, 2, input_dim), 
                              representing the two tokens for each example in the batch.

        Returns:
            torch.Tensor: The logits for classification, with shape (batch_size, output_dim).
                If activation is True, also returns the representation before the final classification layer.
        """
        h = self.get_representation(x) # (batch_size, rep_dim*)
        logits = self.get_logits(h) # (batch_size, output_dim*)
        if activation : return logits, h
        return logits