#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Layers used for the Transformer model
I utilize a bit of unique notation to keep track of the dimensions of tensors. I will follow the following rules in my
code to make the tracking of dimensions as simple as possible.
1. I will use the tensor variable letter as directly from the papers whenever possible.
2. I will use variables that exhibit th applied multiplication as much as possible
3. I will use comments that showcase the computation as well as its resultant dimensions. I will either use one of the
    following notations for my comments:
    * [variable] -> [Dimensions]
    * [computation] -> [computation dimensions] -> [final dimensions]
"""

# Python core libraries

# External libraries
import torch
import torch.nn as nn
from torch import Tensor
import math


class PositionalEncoding(nn.Module):
    """Positional Encoder
    Layer used for obtaining the positional encoding given numerical tokens
    """
    def __init__(self,
                 d_model: int,
                 max_seq_len: int=80):
        super().__init__()
        self.d_model = d_model
        # Create zero matrix of shape max_seq_len x d_model
        PE = torch.zeros(max_seq_len,
                         d_model)
        # Register variable to not be changed by gradients
        # register_buffer used to allow serialization
        PE.requires_grad = False
        # Fill up the matrix
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                # Even dimensions
                sin_val = math.sin(pos / (10_000 ** ((2*i)/d_model)))
                PE[pos, i] = sin_val
                # Odd dimensions
                cos_val = math.cos(pos / (10_000 ** ((2*i)/d_model)))
                PE[pos, i+1] = cos_val
        # PE needs to be a tensor, not a matrix!
        # goes from max_seq_len x d_model -> 1 x max_seq_len x d_model
        PE = PE.unsqueeze(0)
        self.register_buffer('PE',
                             PE)

    def forward(self,
                src: Tensor) -> Tensor:
        '''
        Provide positional embedding tensor, but do not add yet.

        Parameters
        ----------
        src : Tensor
            Input tensor containing tokenized text.
            Expected shape: batch x words+ x d_model

        Returns
        -------
        Tensor
            The positional embedding. This will be the matrix before adding
            to the actual input

        '''
        # Will return Tensor 1 x seq_len x d_model
        return self.PE[:, : src.size(1), :]

# This will not be used in MultiHeadedAttention because we use an optimization trick there.
class Attention(nn.Module):
    def __init__(self,
                 d_m: int,
                 d_k: int,
                 d_v: int=None,
                 optim: str=None,
                 dropout=0.1):
        """Attention Layer
        Parameters
        ----------
        d_m: int
            Input model dimensions
        d_k: int
            Desired query-key dimensions
        d_v: int
            Desired value dimensions. If none, d_k is used.
        optim: str
            Apply matrix optimization. Use None, 'self', 'emb' for no optmization, self-attention optimization, and
            other embedding/memory optimization. If set, only the target application for the optimization is allowed.
        dropout: float
            Dropout probability.
        """
        super().__init__()
        # First d_k outputs correspond to Q
        # Second d_k values correspond to K
        # Last d_v  values correspond to  V
        # [ Query | Key | Value ]
        self.d_v = d_k if d_v is None else d_v

        # Applying optimizations
        if optim == 'self':
            # Self attention
            self.qkv_projection = nn.Linear(d_m, 2*d_k + d_v)
        elif optim == 'emb':
            # Non-self attention, using other embeddings.
            self.q_projection = nn.Linear(d_m, d_k)
            self.kv_projection  = nn.Linear(d_m, d_k + d_v)
        else:
            self.q_projection = nn.Linear(d_m, d_k)
            self.k_projection = nn.Linear(d_m, d_k)
            self.v_projection = nn.Linear(d_m, d_v)


        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.d_k    = d_k
        self.d_m    = d_m
        self.d_v    = d_v
        self.optim  = optim

    def attention(self,
                  Q: Tensor,
                  K: Tensor,
                  V: Tensor,
                  mask: Tensor=None
                  ) -> Tensor:
        """Compute attention
        parameters
        ----------
        Q: Tensor
            Query tensor. Expected shape: batch x n_t x d_k
        K: Tensor
            Key tensor. Expected shape:   batch x n_s x d_k
        V: Tensor
            Value tensor. Expected shape: batch x n_s x d_v
        mask: Tensor
        Working on this. It is incomplete.
        """
        # QK -> batch x n_t x n_s
        QK = (Q @ K.transpose(-1, -2)) / (self.d_k ** (1/2))
        # a -> batch x n_t x d_v
        a  = self.softmax(self.dropout(QK)) @ V
        return a

    def forward(self,
                t: Tensor,
                s: Tensor=None):
        """Attention
        parameters
        ----------
        t: Tensor
            Input to be used as Query. Expected shape is batch x n_q x d_m
        s: Tensor
            Input to be used as Key and Value pair. Expected shape is batch x n_k x d_m
        """

        if self.optim == 'self':
            assert s is None, 'Passed memory/context when not expected. Wrong optmization?'
            QKV = self.qkv_projection(t)
            Q = QKV[:,:,:self.d_k]
            K = QKV[:,:,self.d_k:2*self.d_k]
            V = QKV[:, :, 2*self.d_k:]
        elif self.optim == 'emb':
            KV = self.kv_projection(s)
            Q  = self.q_projection(t)
            K  = KV[:, :, :self.d_k]
            V  = KV[:, :, self.d_k:]
        else:
            Q = self.q_projection(t)
            K = self.k_projection(s)
            V = self.v_projection(s)

        """ Initial matrix shapes
        Q: Tensor
            The Query tensor. It is expected to have shape batch x n_q x d_k
        K: Tensor
            The Key tensor. It is expected to have shape batch x n_k x d_k
        V: Tensor
            The  Value tensor. It is expected to have shape batch x n_k x d_v
        """

        # a -> batch x n_t x d_v
        a = self.attention(Q, K, V)
        return a



class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_m: int,
                 d_k: int,
                 h: int,
                 d_v: int = None,
                 optim: str = None,
                 head_split: bool = True,
                 dropout=0.1):
        """Multi-Headed Attention Layer

        Parameters
        ----------
        d_m: int
            Input model dimensions
        d_k: int
            Desired query-key dimensions
        h: int
            Number of desired attention heads.
        d_v: int
            Desired value dimensions. If None, d_k is used.
        optim: str
            Apply matrix optimization. Use None, 'self', 'emb' for no optmization, self-attention optimization, and
            other embedding/memory optimization. If set, only the target application for the optimization is allowed.
        head_split: bool
            Flag for splitting and concatenating heads later. Only possible if d_k and d_v are divisible by h
        dropout: float
            Dropout probability.
        """
        super().__init__()

        self.d_v = d_k if d_v is None else d_v
        self.head_split = head_split

        if self.head_split:
            '''
            Head split means that we will use d_k/h and d_v/h for the attention output embedding. In this way, when we concatenate
            all of the h head outputs, we will end up with vectors of the original d_v size. This can be helpful in more
            efficient computation, but does decrease the individual head output embedding dimension, so do pay attention
            to that. If we do not use head splitting, then, we will end up with d_v*h embedding dimension, so it is of
            uttermost importance that you keep track of this value, as the embedding may be rather large and computation
            expensive.
            '''
            assert not (d_k % h), f'Head splitting not possible. d_k {d_k} not divisible by h {h}'
            assert not (d_v % h), f'Head splitting not possible. d_k {d_v} not divisible by h {h}'
            self.d_k = d_k // h
            self.d_q = self.d_k # Query embedding must be the same as d_k
            self.d_v = d_v // h
        else:
            # No head splitting. Be careful as the embedding dimensions will be multiplied by h.
            # This is of particular importance as the number of parameters could explode!
            self.d_k = d_k
            self.d_q = self.d_k # Query embedding must be the same as d_k
            self.d_v = d_v

        self.d_m = d_m
        self.h = h
        self.optim = optim
        self.head_split = head_split
        self.dropout = dropout

        # Applying optimizations
        if optim == 'self':
            # Self attention
            # First d_k outputs correspond to Q
            # Second d_k values correspond to K
            # Last d_v  values correspond to  V
            # [ Query | Key | Value ]
            '''For a more indepth explanation, see the else case bellow.
            To extract the Query, Key, and Value, we would need the following:
            1. Slice out Query, Key, and Value
            2. Reshape tensors to have dimensions: b x h x n_q/k x d_v,
            '''
            self.qkv_projection = nn.Linear(self.d_m, self.h * (self.d_q + self.d_k + self.d_v))
        elif optim == 'emb':
            # Non-self attention, using other embeddings.
            self.q_projection = nn.Linear(self.d_m, self.h * self.d_q)
            '''For a more indepth explanation, see the else case bellow.
            To extract the Key and Values, we would need to:
            1. Slice out key and value.
            2. Reshape tensors to end up with dimensions: b x h x n_k/v x d_k/v,
            '''
            self.kv_projection = nn.Linear(self.d_m, self.h * (self.d_k + self.d_v))
        else:
            # No optimizations. Query, Key, and Values will be computated through 3 different matrix multiplications.
            '''Multiheaded computation
            To increase the efficency of computation, it is possible to merge all the heads into a single, large layer.
            This layer will be made up of a matrix of d_m x h * d_(q,k,v). This would imply that we use a single matrix
            multiplication, speeding up computation. To extract the individual heads, we could do two things:
            * simply extract them following this diagram: [h_1 | h_2 | ... | h_h], slicing as we need. The down side is 
                that we would need to concatenate later for the attention computation.
            * Reshape the output using *.reshape(*.shape[0], -1, h, d_v).transpose(1, 2) -> b x h x n_q/k x d_v,
                leaving the tensor ready for the attention computation.
            '''
            self.q_projection = nn.Linear(self.d_m, self.h * self.d_q) # d_q = d_k
            self.k_projection = nn.Linear(self.d_m, self.h * self.d_k)
            self.v_projection = nn.Linear(self.d_m, self.h * self.d_v)
        # Output linear projection from Vaswani paper.
        self.output  = nn.Linear(self.h * self.d_v, self.d_m)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout_q = nn.Dropout(self.dropout)
        self.dropout_k = nn.Dropout(self.dropout)
        self.dropout_v = nn.Dropout(self.dropout)
        self.dropout_o = nn.Dropout(self.dropout)



    def attention(self,
                  Q: Tensor,
                  K: Tensor,
                  V: Tensor,
                  mask: Tensor=None
                  ) -> Tensor:
        """Compute attention
        parameters
        ----------
        Q: Tensor
            Query tensor. Expected shape: batch x n_tgt x d_k or  batch x h x n_tgt x d_k
        K: Tensor
            Key tensor. Expected shape:   batch x n_src x d_k or batch x h x n_src x d_k
        V: Tensor
            Value tensor. Expected shape: batch x n_src x d_v or batch x h x n_src x d_v
        mask: Tensor
            Masking for Query/Key values.
            It is expected in shape       batch x n_tgt x n_src or batch x 1 x n_tgt x n_src
            The Tensor will have -infty for values to be masked, and 0 for values that will not be masked.
        """
        # Make sure all tensors are found in the same device!
        assert Q.device == K.device == V.device, f"Query, Key, and/or Value not in same device {Q.device.type} vs {K.device.type} vs {V.device.type}"

        if mask is None:
            # This mask permits all information. a + 0 = a
            # mask -> 1 x 1 x n_tgt x n_src
            mask = torch.zeros(1, 1, Q.shape[-2], K.shape[-2]).to(Q.device)
        # QK -> batch x h x n_t x n_s
        QK = (Q @ K.transpose(-1, -2) + mask) / (self.d_k ** (1/2))
        # a -> batch x h x n_t x d_v
        a  = self.softmax(QK) @ V
        return a

    def forward(self,
                src: Tensor,
                emb: Tensor=None,
                mask: Tensor=None
                ) -> Tensor:
        if emb is None:
            emb = src

        '''Initial shapes
        src  -> b x n_q x  d_m
        emb  -> b x n_kv x d_m
        mask -> b x 1 x n_q x n_kv 
        '''


        # First, collect Q, K, and V
        if self.optim == 'self': # Self-attention optimization. External embeddings are ignored, as this is self-attn
            # QKV -> b x n_q x h * (d_q + d_k + d_v)
            QKV = self.qkv_projection(src)
            # Q -> b x n_q x h * d_q
            Q = QKV[:,:,:self.h * self.d_q]
            # Q (reshape)-> b x n_q x h x d_q (transpose)-> b x h x n_q x d_q
            Q = Q.reshape(Q.shape[0], -1, self.h, self.d_q).transpose(1, 2)
            # K -> b x n_q x h * d_k
            K = QKV[:,:,self.h * self.d_q: self.h * (self.d_q + self.d_k)]
            # K (reshape)-> b x n_q x h x d_k (transpose)->
            K = K.reshape(K.shape[0], -1, self.h, self.d_k).transpose(1, 2)
            # V -> b x n_q x h * d_v
            V = QKV[:,:,self.h * (self.d_q + self.d_k):]
            # V (reshape)-> b x n_q x h x d_v (transpose)-> b x h x n_q x d_v
            V = V.reshape(V.shape[0], -1, self.h, self.d_v).transpose(1, 2)

        elif self.optim == 'emb': # Other embedding optimization
            # Q -> b x n_q x h * d_q=k
            Q = self.q_projection(src)
            # Q (reshape)-> b x n_q x h x d_q (transpose)-> b x h x n_q x d_q
            Q = Q.reshape(Q.shape[0], -1, self.h, self.d_q).transpose(1, 2)
            # KV -> b x n_kv x h * (d_k + d_v)
            KV = self.kv_projection(emb)
            # K -> b x n_kv x h * d_k
            K = KV[:,:,:self.h * self.d_k]
            # K (reshape)-> b x n_kv x h x d_k  (transpose)-> b x h x n_kv x d_k
            K = K.reshape(K.shape[0], -1, self.h, self.d_k).transpose(1, 2)
            # V -> b x n_kv x h * d_v
            V = KV[:,:,self.h * self.d_k:]
            # V (reshape)-> b x n_kv x h x d_v  (transpose)-> b x h x n_kv x d_v
            V = V.reshape(V.shape[0], -1, self.h, self.d_v).transpose(1, 2)
        else:
            '''Applying attention without particular optimizations. Good for understanding code.'''
            # Q -> b x n_q x h * d_q=k
            Q = self.q_projection(src)
            # Q (reshape)-> b x n_q x h x d_q (transpose)-> b x h x n_q  x d_q
            Q = Q.reshape(Q.shape[0], -1, self.h, self.d_q).transpose(1, 2)
            # K -> b x n_kv x h * d_k=q
            K = self.k_projection(emb)
            # K (reshape)-> b x n_kv x h x d_k=q (transpose)-> b x h x n_kv x d_k=q
            K = K.reshape(K.shape[0], -1, self.h, self.d_k).transpose(1, 2)
            # V -> b x n_kv x h * d_v
            V = self.v_projection(emb)
            # V (reshape)-> b x n_kv x h x d_v (transpose)-> b x h x n_kv x d_v
            V = V.reshape(V.shape[0], -1, self.h, self.d_v).transpose(1, 2)
        # Apply Dropout
        Q = self.dropout_q(Q)
        K = self.dropout_k(K)
        V = self.dropout_v(V)
        # a -> b x h x n_q x d_v
        a = self.attention(Q, K, V, mask)
        # a (transpose)-> b x n_q x h x d_v (view/reshape)-> b x n_q x h * d_v
        a = a.transpose(1, 2).contiguous().view(a.shape[0], -1, self.h * self.d_v)
        # output -> b x n_q x d_m
        output = self.output(a)
        # Last dropout layer
        output = self.dropout_o(output)
        return output

