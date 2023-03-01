#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transformer Model in Pytorch

This is yet another transformer model programmed in Pytorch.

Code is organized in 3 sections:
* Encoder Layers
    Contains code for Encoder layers and Transformer Encoder module
* Decoder Layers
    Contains code for Decoder layers and Transformer Decoder module
* Complete Transformers
    Implementations of full transformers. Currently, TranslatorTransformer uses both Transformer Modules.

@author: Jose E. Aguilar Escamilla
"""

# Python Core
from typing import Union, Callable, Tuple

# External Libraries (AI)
import torch
import torch.nn as nn
from torch import Tensor

# Internal Libraries
from .layers import PositionalEncoding, MultiHeadAttention

''' Encoder Layers '''

class EncoderLayer(nn.Module):
    def __init__(self,
                 d_m: int,
                 d_k: int,
                 h: int,
                 d_ff: int,
                 d_v: int=None,
                 head_split: bool=True,
                 dropout: float=0.1):
        super().__init__()
        # Multiheaded Self Attention
        self.mha = MultiHeadAttention(d_m       =d_m,
                                      d_k       =d_k,
                                      h         =h,
                                      d_v       =d_v,
                                      optim     ='self',
                                      head_split=head_split,
                                      dropout   =dropout,)
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

    def forward(self,
                src: Tensor,
                mask_src: Tensor=None,
                ) -> Tensor:
        """Re-encode input embedding using attention
        Parameters
        ----------
        src: Tensor
            Input embedding to re-encode. Expected to be in shape b x n_q x d_m
        mask_src: Tensor
            Masking (often padding mask) to selectively ignore information by the attention modules. since this encoder
            module uses self-attention, It is expected to be in shape b x 1 x n_q x n_q.
        """

        # src_mha -> b x n_q x d_m
        src_mha = self.mha(src=src,
                           emb=src, # Counts for both Key and Value.
                           mask=mask_src)
        # src_add_norm -> b x n_q x d_m
        src_add_norm = self.norm1(src + src_mha)
        # src_ff -> b x n_q x d_m
        src_ff = self.ff(src_add_norm)
        # src_ff_add_norm -> b x n_q x d_m
        src_ff_add_norm = self.norm2(src_add_norm + src_ff)

        return src_ff_add_norm


class TransformerEncoder(nn.Module):
    def __init__(self,
                 d_m: int,
                 d_k: int,
                 h: int,
                 N: int,
                 d_ff: int,
                 max_seq_len: int,
                 embeder: Callable[[Tensor], Tensor],
                 d_v: int=None,
                 head_split: bool=True,
                 dropout: float=0.1):
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
        max_seq_len: int
            Maximum expected length for sentences or inputs. Important for allowing batch processing.
            It will be expected that some certain token will be used for padding when masking.
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
        self.max_seq_len = max_seq_len
        self.embeder = embeder
        self.d_v = d_v
        self.head_split = head_split
        self.dropout = dropout


        # Positional Encoding to be added later.
        self.PE = PositionalEncoding(d_m,
                                     max_seq_len)
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
            Input tokenized data. Expected to be in shape b x max_seq_len
        mask_src: Tensor
            Masking (often padding mask) to selectively ignore information in attention. Expected to be in shape
            batch x 1 x max_seq_len x max_seq_len.
            This mask is applied to a multiheaded self attention layer. 0s allow data and -infty masks information.
        """

        # src_emb -> b x max_seq_len=n_q x d_m
        src_emb = self.embeder(src)
        PE = self.PE(src_emb).to(src_emb.device.type)
        src_PE = src_emb + PE
        # src_enc -> b x n_q x d_m
        src_enc = src_PE # Prepare for loop
        for encoder in self.encoders:
            src_enc = encoder(src_enc, mask_src)
        return src_enc

''' Decoder Layers'''

class DecoderLayer(nn.Module):
    def __init__(self,
                 d_m: int,
                 d_k: int,
                 h: int,
                 d_ff: int,
                 d_v: int = None,
                 head_split: bool = True,
                 dropout: float = 0.1
                 ):
        super().__init__()

        # First multi-headed attention module. Will use masking heavily!
        self.masked_mha = MultiHeadAttention(d_m=d_m,
                                             d_k=d_k,
                                             h=h,
                                             d_v=d_v,
                                             optim='self',
                                             head_split=head_split,
                                             dropout=dropout)
        # First Normalization layer
        self.norm1 = nn.LayerNorm(d_m)
        # Second attention module. Combines the encoder embeddings.
        self.emb_mha = MultiHeadAttention(d_m=d_m,
                                          d_k=d_k,
                                          h=h,
                                          d_v=d_v,
                                          optim='emb',
                                          head_split=head_split,
                                          dropout=dropout)
        # Second normalization layer
        self.norm2 = nn.LayerNorm(d_m)
        # Feed-forward ReLU neural network
        self.ff = nn.Sequential(nn.Linear(d_m, d_ff),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_m),
                                nn.Dropout(dropout))
        # Thirs normalization layer
        self.norm3 = nn.LayerNorm(d_m)

    def forward(self,
                tgt:      Tensor,
                src_emb:  Tensor,
                mask_pad_fwd: Tensor = None,
                mask_tgt_src: Tensor = None
                ) -> Tensor:
        """Decode embedding
        parameters
        -----------
        tgt: Tensor
            Encoder input to use for decoding. Expected to be in shape b x n_q=n_tgt x d_m
        src_emb: Tensor
            Embeddings collected from an embedder/encoder. Expected in shape b x n_k=n_src x d_m
        mask_pad_fwd: Tensor
            Padding and forward information flow mask. Used in the first attention layer. Expected shape b x 1 x n_tgt x n_tgt
        mask_tgt_src: Tensor
            Padding mask for src and tgt sentences. Expected shape b x 1 x n_tgt x n_src
        """
        # tgt_mask_mha -> b x n_q x d_m
        tgt_mask_mha = self.masked_mha(tgt,
                                       mask=mask_pad_fwd)
        # tgt_add_norm -> b x n_q x d_m
        tgt_add_norm = self.norm1(tgt + tgt_mask_mha)
        # tgt_add_norm -> b x n_q x d_m
        tgt_emb_mha  = self.emb_mha(tgt_add_norm,
                                    src_emb,
                                    mask=mask_tgt_src)
        tgt_mha_add_norm = self.norm2(tgt_emb_mha + tgt_add_norm)
        # tgt_ff -> b x n_q x d_m
        tgt_ff = self.ff(tgt_mha_add_norm)
        tgt_ff_add_norm = self.norm3(tgt_ff + tgt_mha_add_norm)
        # Theoretically, this retained the original shape!
        return tgt_ff_add_norm

class TransformerDecoder(nn.Module):
    def __init__(self,
                 d_m: int,
                 d_k: int,
                 h: int,
                 N: int,
                 d_ff: int,
                 d_vocab: int,
                 max_seq_len: int,
                 embeder: Callable[[Tensor], Tensor],
                 d_v: int=None,
                 head_split: bool=True,
                 dropout: float=0.1):
        super().__init__()
        self.d_m = d_m
        self.d_k = d_k
        self.h = h
        self.N = N
        self.d_ff = d_ff
        self.d_vocab = d_vocab
        self.max_seq_len = max_seq_len
        self.embeder = embeder
        self.d_v = d_v
        self.head_split = head_split
        self.dropout = dropout

        # Positional Encoding to be added later.
        self.PE = PositionalEncoding(d_m,
                                     max_seq_len)
        # Decoder stacked layers
        # Create layers
        decoder_layers = [DecoderLayer(d_m,
                                       d_k,
                                       h,
                                       d_ff,
                                       d_v,
                                       head_split,
                                       dropout)
                          for _ in range(N)]
        self.decoders = nn.ModuleList(decoder_layers)
        # Output probability distribution. d_vocab refers to the number of words in a vocabulary
        self.prob_dist = nn.Sequential(
            nn.Linear(d_m, d_vocab),
            nn.Softmax(dim=-1)
        )

    def forward(self,
                tgt:      Tensor,
                src_enc:  Tensor,
                mask_pad_fwd: Tensor = None,
                mask_tgt_src: Tensor = None
                ) -> Tensor:
        """Decode encoded information
        parameters
        ----------
        tgt: Tensor
            Decoder input. Expected in shape batch x max_seq_len
        src_enc: Tensor
            Encoder embeded information. Expected in shape batch x max_seq_len x d_m
        mask_pad_fwd: Tensor
            padding/forward flow mask. Applied to first attention layer in decoder module. Expected b x 1 x n_tgt x n_tgt
        mask_tgt_src: Tensor
            src/tgt padding mask. Applied in second attention layer in decoder module. Expected b x 1 b n_tgt x n_src
        """
        # tgt_emb -> b x n_q x d_m
        tgt_emb = self.embeder(tgt)
        PE = self.PE(tgt_emb)
        tgt_PE = tgt_emb + PE
        # tgt_enc -> b x n_q x d_m
        tgt_enc = tgt_PE
        for decoder in self.decoders:
            tgt_enc = decoder(tgt_enc,
                              src_enc,
                              mask_pad_fwd=mask_pad_fwd,
                              mask_tgt_src=mask_tgt_src,)
        # tgt_prob_dist -> b x n_q x d_vocab
        tgt_prob_dist = self.prob_dist(tgt_enc)
        # This probability distribution is the probability of the queried word w.r.t. the other vocab
        return tgt_prob_dist

''' Full Transformers '''


class TranslatorTransformer(nn.Module):
    def __init__(self,
                 d_m: int,
                 d_k: int,
                 h: int,
                 N: int,
                 d_ff: int,
                 d_vocab: int,
                 max_seq_len: int,
                 embeder: Callable[[Tensor], Tensor],
                 d_v: int = None,
                 head_split: bool = True,
                 dropout: float = 0.1,
                 ):
        super().__init__()
        self.d_m = d_m
        self.d_k = d_k
        self.h = h
        self.N = N
        self.d_ff = d_ff
        self.d_vocab = d_vocab
        self.max_seq_len = max_seq_len
        self.embeder = embeder
        self.d_v = d_v
        self.head_split = head_split
        self.dropout = dropout
        # Add upper triangular matrix for forward mask

        self.encoder = TransformerEncoder(d_m,
                                          d_k,
                                          h,
                                          N,
                                          d_ff,
                                          max_seq_len,
                                          embeder,
                                          d_v,
                                          head_split,
                                          dropout)
        self.decoder = TransformerDecoder(d_m,
                                          d_k,
                                          h,
                                          N, d_ff,
                                          d_vocab,
                                          max_seq_len,
                                          embeder,
                                          d_v,
                                          head_split,
                                          dropout)
        # Do we need to register a buffer?

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_pre_mask: Tensor = None,
                tgt_pre_mask: Tensor = None,
                flow: str = 'all',
                ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Language Transformer
        parameters
        ----------
        src: Tensor
            Input sentence to encode. Expected in shape b x max_seq_len
        tgt: Tensor
            Input sentence to decode. Expected in shape b x max_seq_len. Often this input is the accumulation of
            previous decoder outputs.
        src_pre_mask: Tensor
            Padding mask to be applied in encoder (and decoder). Expected to be in shape b x max_seq_len with dtype=bool.
        tgt_pre_mask: Tensor
            Padding mask to be applied in decoder. Expected to be in shape b x max_seq_len with dtype=bool
        flow: str
            Flag for selecting type of flow. Three options:
            * encoder
                Flow only through encoder. Model returns encoded inputs, ignoring tgt and tgt_mask.
            * decoder
                Flow only through encoder. Model returns next word for tgt sequence. src is expected to be encoded src,
                and it is expected in shape b x max_seq_len x d_m. src_mask is expected as b x max_seq_len
            * all
                Flow through both encoder and decoder. src and tgt expected to be b x max_seq_len, being ordinal tokens.
        """

        # Make sure Tensors located in same device
        assert src.device == tgt.device, f"src and tgt Tensors are not on the same device: {src.device.type} vs {tgt.device.type}"

        # Prepare pre_masks for mask computation. Convert bool Tensor into tensor where 0 lets information flow, and
        # -infty blocks the information
        # Create non-empty masks
        if src_pre_mask is None:
            src_pre_mask = torch.zeros(src.shape).bool().to(src.device.type)
        if tgt_pre_mask is None:
            tgt_pre_mask = torch.zeros(tgt.shape).bool().to(src.device.type)

        # src_mask -> b x n_src x 1
        src_mask = src_pre_mask.float()
        src_mask[src_pre_mask] -= torch.inf
        src_mask = src_mask.unsqueeze(-1)
        # tgt_mask -> b x n_tgt x 1
        tgt_mask = tgt_pre_mask.float()
        tgt_mask[tgt_pre_mask] -= torch.inf
        tgt_mask = tgt_mask.unsqueeze(-1)

        '''Encoder mask matrix computation'''
        # pad_mask -> b x 1 x n_src x n_src
        mask_pad_src = (src_mask @ src_mask.transpose(-1, -2)).unsqueeze(1)

        '''Decoder Mask matrix computation'''
        # Padding mask to be used in decoder first masked self-attention layer.
        # mask_fwd_tgt -> b x 1 x n_tgt x n_tgt
        mask_fwd_tgt = torch.zeros(tgt.shape[0], 1, tgt.shape[1], tgt.shape[1]) - torch.inf # a -infty matrix
        mask_fwd_tgt = mask_fwd_tgt.triu(diagonal=1).to(src.device.type) # Upper triangular matrix excluding diagonal
        # mask_pad_tgt -> b x 1 x n_tgt x n_tgt
        mask_pad_tgt = (tgt_mask @ tgt_mask.transpose(-1, -2)).unsqueeze(1)
        # mask_pad_fwd_tgt -> b x 1 x n_tgt x n_tgt
        mask_pad_fwd_tgt = mask_fwd_tgt + mask_pad_tgt
        # mask_pad_tgt_src -> b x 1 x n_tgt x n_src
        mask_pad_tgt_src = (tgt_mask @ src_mask.transpose(-1, -2)).unsqueeze(1)

        assert flow in ['encoder', 'decoder', 'all'], f'Invalid targeted flow {flow}. Only encoder, decoder, or all.'

        if flow == 'all':
            # Encode and Decode information
            src_enc = self.encoder(src,
                                   mask_pad_src)
            tgt_prob_dist = self.decoder(tgt,
                                         src_enc,
                                         mask_pad_fwd=mask_pad_fwd_tgt,
                                         mask_tgt_src=mask_pad_tgt_src)
            return tgt_prob_dist, src_enc

        elif flow == 'encoder':
            # Encode only
            src_enc = self.encoder(src,
                                   mask_src=mask_pad_src)
            return src_enc

        elif flow == 'decoder':
            # Decode only
            tgt_prob_dist = self.decoder(tgt,
                                         src,
                                         mask_pad_fwd=mask_pad_fwd_tgt,
                                         mask_tgt_src=mask_pad_tgt_src)
            return tgt_prob_dist
