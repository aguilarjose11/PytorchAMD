#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Attention Model (AM-D)

@author: Jose E. Aguilar Escamilla
@email: jose.efraim.a.e@gmail.com
"""


# Python Libraries
from typing import Callable, Union, Tuple

# External Libraries
import torch
from torch import Tensor
import torch.nn as nn

from .layers import MultiHeadAttention, PDAttention


''' Encoder Layers '''
class EncoderLayer(nn.Module):
    def __init__(self,
                 d_m: int,
                 d_k: int,
                 h: int,
                 d_ff: int,
                 d_v: int = None,
                 head_split: bool = True,
                 dropout: float = 0.1):
        """Encode graph nodes.
        Creates an encoder layer that will encode graph nodes form an input combinatorial graph problem.
        parameters
        ----------
        d_m: int
            Model dimensions to be used within the layer.
        d_k: int
            Query/Key dimensions to allow comparison of query and key.
        h: int
            Number of heads for attention layers.
        d_ff: int
            Hidden neurons for feed forward neural network.
        d_v: int
            Value dimension for passing information after query/key comparison. If none, uses d_k
        head_split: bool
            Whether to split the heads of MHA across d_k/d_v or not. Can permit computational efficency at the cost of
            reducing the latent dimension for MHA.
        dropout: float
            Regularization value for dropout. Probability that an input is converted to a 0.
        """
        super().__init__()
        # Multiheaded Self Attention
        self.mha = MultiHeadAttention(d_m=d_m,
                                      d_k=d_k,
                                      h=h,
                                      d_o=d_m,
                                      d_v=d_v,
                                      optim='self',
                                      head_split=head_split,
                                      dropout=dropout,)
        # first normalization layer
        self.norm1 = nn.LayerNorm(d_m)

        # ReLU Feed forward neural network
        self.ff = nn.Sequential(nn.Linear(d_m, d_ff),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_m),
                                nn.Dropout(dropout))

        # Second normalization layer
        self.norm2 = nn.LayerNorm(d_m)

        # Saving layer configuration
        self.d_m = d_m
        self.d_k = d_k
        self.h = h
        self.d_ff = d_ff
        self.d_v = d_v
        self.hed_split = head_split
        self.dropout = dropout

    def forward(self,
                src: Tensor,
                mask_src: Tensor=None,
                ) -> Tensor:
        """Re-encode input embedding using attention
        Parameters
        ----------
        src: Tensor
            Input embedding to re-encode. Expected to be in shape b x nodes x d_m. The graph is expected to have already
            been linearly-projected.
        mask_src: Tensor
            Masking (often invalid nodes) to selectively ignore information by the attention modules. since this encoder
            module uses self-attention, It is expected to be in shape b x 1 x nodes x nodes.
        """

        # src_mha -> b x nodes x d_m
        src_mha = self.mha(src=src,
                           emb=src, # Counts for both Key and Value.
                           mask=mask_src)
        # src_add_norm -> b x nodes x d_m
        src_add_norm = self.norm1(src + src_mha)
        # src_ff -> b x nodes x d_m
        src_ff = self.ff(src_add_norm)
        # src_ff_add_norm -> b x nodes x d_m
        src_ff_add_norm = self.norm2(src_add_norm + src_ff)

        return src_ff_add_norm

# Remove PositionalEncoding
class TransformerEncoder(nn.Module):
    def __init__(self,
                 d_m: int,
                 d_k: int,
                 h: int,
                 N: int,
                 d_ff: int,
                 n_nodes: int,
                 embeder: Union[nn.Module, int],
                 d_v: int = None,
                 head_split: bool = True,
                 dropout: float = 0.1):
        """Transformer Encoder Module
        This Transformer module will build the needed layers for creating a transformer encoder.

        parameters
        ----------
        d_m: int
            Dimensions to use with model embedding. This dimension value should be the same as the
            one used for first embedding the inputs.
        d_k: int
            Key dimensions for attention modules.
        h: int
            Number of heads to use for all layers
        N: int
            Number of encoder layers
        d_ff: int
            Feed forward neural network hidden layer dimensions
        n_nodes: int
            Number of nodes to solve for.
        embeder: Callable[[Tensor], Tensor] | int
            A callable function or int that embeds a given input. The original paper for Attention Model simply uses a
            linear transformation for this based on the dimensions and information of the nodes. If embeder is an int,
            the module will assume the value is the individual node dimensions and will create a linear projection to
            convert the node dimensions into d_m. For example, if each node has 3 pieces of information, passing 3 to
            the parameter will create a linear projection described by a 3 x d_m matrix.
        d_v: int
            Value dimensions for attention modules. Often it is made to equal d_k. This will also be
            the embedding dimensions for each output of the attention layers.
        head_split: bool
            Whether to distribute the information embeddings across the heads, or treat each head as
            its own. Note that if this is false, the number of parameters will be multiplied by h.
        dropout: float
            Dropout for regularization
        """
        super().__init__()

        self.d_m = d_m
        self.d_k = d_k
        self.h = h
        self.N = N
        self.d_ff = d_ff
        self.n_nodes = n_nodes
        self.embeder = embeder if type(embeder) is not int else nn.Linear(embeder, d_m)
        self.d_v = d_v
        self.head_split = head_split
        self.dropout = dropout
        # We do not require positional encoding for graph problems.
        # Encoder stacked layers
        # Create layers
        encoder_layers = [EncoderLayer(d_m,
                                       d_k,
                                       h,
                                       d_ff,
                                       d_v,
                                       head_split,
                                       dropout)
                          for _ in range(N)]
        # Then, stack them up! We use ModuleList instead of Sequential since we have to pass the mask
        self.encoders = nn.ModuleList(encoder_layers)

    def forward(self,
                src:  Tensor,
                mask_src: Tensor = None,
                ) -> Tensor:
        """Compute input embedding using attention
        parameters
        ----------
        src: Tensor
            Input graph nodes. Expected to be in shape b x nodes x d_nodes, where d_nodes is the information related to
            each individual node.
        mask_src: Tensor
            Masking (often padding mask) to selectively ignore information in attention. Expected to be in shape
            batch x 1 x nodes x nodes.
            This mask is applied to a multiheaded self attention layer. 0s allow data and -infty masks information.
        """

        # src_emb -> b x nodes x d_m
        src_emb = self.embeder(src)
        # src_enc -> b x nodes x d_m
        src_enc = src_emb # Prepare for loop
        for encoder in self.encoders:
            src_enc = encoder(src_enc, mask_src)
        return src_enc

''' Decoder Layers '''

# TODO: Convert this to the AM/AM-D decoder
class DecoderLayer(nn.Module):
    def __init__(self,
                 d_m: int,
                 d_c: int,
                 d_k: int,
                 h: int,
                 d_ff: int,
                 c: float = 10.,
                 d_v: int = None,
                 head_split: bool = True,
                 dropout: float = 0.1
                 ):
        """ Decode information to create a probability distribution over which node is most optimal.
        parameters
        ----------
        d_m: int
            Embedding dimension generated by the encoder.
        d_c: int
            Input graph context dimension.
        d_k: int
            Query/Key comparison dimension.
        h: int
            Number of heads for MHA layer.
        d_ff: int
            Dimensions of hidden layer of ReLU feed forward network.
        c: float
            Clipping value for logits.
        d_v:
            Dimensions of the Value matrix. If None, will use d_m
        head_split: bool
            Flag for splitting the dimensions of d_v/k among the heads. This helps improve computational performance
            at the cost of a reduced latent space dimensionality.
        dropout: float
            Regularization dropout.

        """
        super().__init__()
        self.d_m = d_m
        self.d_c = d_c
        self.d_k = d_k
        self.h = h
        self.d_ff = d_ff
        self.c = c
        self.d_v = d_v if not None else d_m
        self.head_split = head_split
        self.dropout = dropout

        # First multi-headed attention module. Will use masking heavily!
        self.emb_mha = MultiHeadAttention(d_m=d_m,
                                          d_k=d_k,
                                          h=h,
                                          d_o=d_c,
                                          d_c=d_c,
                                          d_v=d_v,
                                          optim='emb',
                                          head_split=head_split,
                                          dropout=dropout)
        # Attention layer for the creation of a probability distribution.
        self.pda = PDAttention(d_m=d_m,
                               d_c=d_c,
                               d_k=d_k,
                               c=c,
                               dropout=dropout)

    def forward(self,
                ctxt:     Tensor,
                src_emb:  Tensor,
                mask_graph: Tensor = None,
                return_node_idx: bool = False
                ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Decode embedding
        parameters
        -----------
        ctxt: Tensor
            Graph context containing information about the current state of the environment.
            Expected in shape b x 1 x d_c
        src_emb: Tensor
            Embeddings collected from an embeder/encoder.
            Expected in shape b x nodes x d_m
        mask_graph: Tensor
            Graph masking. Often applied to remove unreachable nodes or already visited ones.
            Expected shape b x 1 x nodes. Will be unsqueezed to b x 1 x 1 x nodes to permit multi-headed masking.
        return_node_idx: Tensor
            Flag for returning the argmax of the probability distribution. This will return not only the probability
            distribution, but also the indices of the node that was selected by the model.

        """
        # tgt_mask_mha -> b x 1 x d_c
        tgt_mask_mha = self.emb_mha(src=ctxt,
                                    emb=src_emb,
                                    mask=mask_graph.unsqueeze(1))
        # tgt_add_norm -> b x 1 x nodes
        tgt_emb_mha = self.pda(tgt_mask_mha,
                                src_emb,
                                mask=mask_graph)
        # Will return a probability distribution.
        if return_node_idx:
            # The argmax vector will have shape b x 1.
            return tgt_emb_mha, tgt_emb_mha.argmax(-1)
        return tgt_emb_mha

# There will not be a TransformerDecoder, since the layer is enough.


class AttentionModel(nn.Module):
    """Dynamic Attention Model
    """
    def __init__(self,
                 d_m: int,
                 d_c: int,
                 d_k: int,
                 h: int,
                 N: int,
                 d_ff: int,
                 n_nodes: int,
                 embeder: Union[nn.Module, int],
                 d_v: int = None,
                 c: float = 10.,
                 head_split: bool = False,
                 dropout: float = 0.1,
                 use_graph_emb: bool = True,
                 batches: int=None):
        """
        

        Parameters
        ----------
        d_m: int
            Model dimensions for embeddings.
        d_c: int
            Context dimensions.
        d_k: int
            Dimensions of key.
        h: int
            Number of attention heads.
        N: int
            Number of encoder layers.
        d_ff: int
            Number of encoder hidden layer neural network dimensions.
        n_nodes: int
            Number of nodes of problem to work with.
        embeder: Union[nn.Module, int]
            Input emedding for embeding module. If int, it creates a linear projection.
        d_v: int
            Dimensions for value matrix.
        c: float
            Clipping value for probability calculation of decoder.
        head_split: bool
            Whether to split d_v/k among heads to improve computation cost at the expense of lowered latent space.
        dropout: float
            Regularization value.
        use_graph_emb: bool
            Flag for indicating whether to calculate the graph embedding (avg from original paper), thus adding d_m to
            d_c.
        batches: int
            Number of batches to expect.

        """
        super().__init__()

        self.d_m = d_m
        # Increment the dimension by d_m, as applying the graph embedding to the context increases its size by d_m.
        self.d_c = d_c + d_m if use_graph_emb else 0
        self.d_k = d_k
        self.h = h
        self.N = N
        self.d_ff = d_ff
        self.n_nodes = n_nodes
        self.embeder = embeder
        self.d_v = d_v
        self.c = c
        self.head_split = head_split
        self.dropout = dropout
        self.use_graph_emb = use_graph_emb
        self.batches = batches
        # Use placeholder batch
        batches = 1 if batches is None else batches
        # This buffer will contain the computed graph embeddings, allowing iterative generation of a path.
        # graph_emb -> batches x n_nodes x d_m
        self.register_buffer('graph_emb', torch.rand(batches, n_nodes, d_m))

        self.encoder = TransformerEncoder(d_m=d_m,
                                          d_k=d_k,
                                          h=h,
                                          N=N,
                                          d_ff=d_ff,
                                          n_nodes=n_nodes,
                                          embeder=embeder,
                                          d_v=d_v,
                                          head_split=head_split,
                                          dropout=dropout)
        self.decoder = DecoderLayer(d_m=d_m,
                                    d_c=self.d_c,
                                    d_k=d_k,
                                    h=h,
                                    d_ff=d_ff,
                                    c=c,
                                    d_v=d_v,
                                    head_split=head_split,
                                    dropout=dropout)

    
    def forward(self, 
                graph: Tensor,
                ctxt: Tensor,
                mask_emb_graph: Tensor,
                mask_dec_graph: Tensor,
                reuse_embeding: bool=False
                ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Parameters
        ----------
        graph: Tensor
            Graph input. Expected in shape b x nodes x d_info. d_info is the initial information available in graphs.
            For example, in the 2D VRP problem, we would expect for a single node to have dimensions 3: x, y, and
            required material.
        ctxt: Tensor
            Context vector. It often is the current environment state. Expected in shape b x 1 x d_c
        mask_emb_graph: Tensor
            Embedder graph mask. Expected in shape b x 1 x nodes. Will be unsqueezed for MHA and multiplied to obtain
            shape b x 1 x nodes x nodes. Used for masking certain nodes from affecting the node embeddings of others.
        mask_dec_graph: Tensor
            Decoder graph mask. Expected in shape b x 1 x nodes. Used to ignore particular node embeddings.
            Will be passed as is.
        reuse_embeding: bool
            Flag to indicate that the previously calculated embeddings are to be used. This will not flow the graph
            data through the embeder, using buffer graph_emb instead.

        """
        device = list(self.parameters())[0].device
        if not reuse_embeding:
            # Compute mask for embedder's self-MHA.
            # We will first apply the logical or operator. Using dot product produces an and operator.
            # mask_emb -> b x nodes x nodes (unsqueeze)-> b x 1 x nodes x nodes
            mask_emb_bool = torch.logical_or(mask_emb_graph.transpose(-1, -2), mask_emb_graph).unsqueeze(1)
            # Create zero mask of shape b x 1 x nodes x nodes
            mask_emb = torch.zeros(mask_emb_bool.shape)
            # Replace all True mask values for -inf
            mask_emb[mask_emb_bool] = -torch.inf
            mask_emb = mask_emb.to(device)

            # Compute node new node embeddings
            # src_emb -> b x nodes x d_m
            src_emb = self.encoder(graph, mask_emb)
            # Figure out whether to replace values or entire buffer. ngl, this is for my own practice. This isn't needed
            if self.batches is not None:
                # Replace the graph embeddings with the newly calculated, maintaining the buffer shape
                self.graph_emb[:] = src_emb
            else:
                # The number of batches encoded before is different. Replacing the entire Tensor.
                # src_emb -> b x nodes x d_m
                self.graph_emb = src_emb
        else:
            # Re-use previous embeddings!
            # src_emb -> b x nodes x d_m
            src_emb = self.graph_emb
        # Pass through to decoder
        # Compute the mask for decoder's MHA
        # mask_dec -> b x 1 x nodes
        mask_dec = torch.zeros(mask_dec_graph.shape)
        # Replace all True values for negative infinity.
        mask_dec[mask_dec_graph] = -torch.inf
        mask_dec = mask_dec.to(device)
        if self.use_graph_emb:
            # If we are to use avg graph embeddings, then, calculate them and add them to ctxt
            # graph_emb_avg -> b x 1 x d_m
            graph_emb_avg = self.graph_emb.detach().mean(-2).unsqueeze(1)
            # graph_emb_avg -> b x 1 x d_v (+ d_m)
            ctxt = torch.cat((ctxt, graph_emb_avg), dim=-1)
            assert ctxt.shape[-1] == self.d_c, f"The context shape and d_c do not match! ctxt: {ctxt.shape[-1]} vs d_c: {self.d_c}."
        # probability_dist -> b x 1 x nodes
        # selection -> b x 1
        probability_dist = self.decoder(ctxt, src_emb, mask_dec, False)
        # Note that we could use graph[selection] to obtain the actual node. We would use self.graph_emb[selection] to
        # find the graph embeddings, which would be fed into the model for the next iteration.
        return probability_dist.clamp(1e-4)
