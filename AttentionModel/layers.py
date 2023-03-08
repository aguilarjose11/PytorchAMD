#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modules/Layers for the Dynamic Attention Model (AM-D)

@author: Jose E. Aguilar Escamilla
@email: jose.efraim.a.e@gmail.com
"""

# Python Libraries

# External Libraries

import torch
import torch.nn as nn
from torch import Tensor
import functorch

import gudhi as gd
import gudhi.representations
import math


class PDAttention(nn.Module):
    """Return a probability distribution (PD) w.r.t. Key"""

    def __init__(self,
                 d_m: int,
                 d_c: int,
                 d_k: int,
                 c: float = 10.,
                 dropout: float = 0.1):
        """Layer for producing probability distribution w.r.t. Key and Query
        parameters
        ----------
        d_m: int
            Dimensions of the input node embeddings from the encoder. This is important for moving the dimensions to match
            those of the decoder.
        d_c: int
            Input graph context dimension. In original AM paper, for TSP, 3*d_m is used (adding of graph embedding and
            start and end embedded nodes.) VRP uses 2*d_m + 1 (the graph embedding and latest node plus remaining capacity.)
            Overall, this dimension will be the -1 dimension of the input, which often is a horizontal concatenation of
            the state of the environment.  Note that in the decoder, the output from previous MHA module will use dimension
            d_c rather than d_m. This is unique to the AM-D model, as no linear projection is applied to the raw graph
            context when passed to the decoder. See figure 2 from AM-D paper.
        d_k: int
            Query and Key dimension.
        c: float
            Clipping constant to apply before logit computation.
        dropout: float
            Dropout value.
        """
        super().__init__()
        # Part of the encoder embeddings
        self.d_m = d_m
        # Dimension of the graph context.
        self.d_c = d_c
        # Dimension to allow mutual comparison.
        self.d_k = d_k
        self.c = c
        self.dropout = dropout

        # Projections for graph context and node embeddings.
        self.q_projection = nn.Linear(self.d_c, self.d_k)
        self.k_projection = nn.Linear(self.d_m, self.d_k)
        self.dropout_q = nn.Dropout(self.dropout)
        self.dropout_k = nn.Dropout(self.dropout)
        self.tan_h = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,
                ctxt: Tensor,
                emb: Tensor,
                mask: Tensor=None) -> Tensor:
        """Compute logits
        parameters
        ----------
        ctxt: Tensor
            Input Query. Expected in shape batch x 1 x d_c, where d_c is a horizontal concatenation of indicators
            relevant to the problem. Often, it is the concatenation of a graph embedding and environment state statistics.
            Note that in the original Transformer, the MHA output was often linearly transformer back to d_m, which was
            the linear projection applied to the input of the decoder. In the AM-D model, the MHA layer will lienarly
            project its output from d_v to d_c instead to d_m. This is to keep consistency with the original input.
        emb: Tensor
            Input Key. Expected in shape batch x nodes x d_m
        mask: Tensor
            Masking on nodes that should not be included in probability distribution. Expected in shape
            batch x 1 x nodes
        return
        ------
        p: Tensor
            A Tensor containing the probability distribution with respect to every node that was made available to the
            encoder. Expected shape of batch x 1 x nodes
        """
        # Q -> batch x 1 x d_k
        Q = self.q_projection(ctxt)
        Q = self.dropout_q(Q)
        # K -> batch x nodes x d_k
        K = self.k_projection(emb)
        K = self.dropout_k(K)
        # u -> batch x 1 x nodes
        u = (Q @ K.transpose(-2, -1)) / (self.d_k ** (1/2))
        # u -> batch x 1 x nodes
        u = (self.c * self.tan_h(u)) + mask if mask is not None else 0
        # p -> batch x 1 x nodes
        p = self.softmax(u.nan_to_num()).nan_to_num()
        return p


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_m: int,
                 d_k: int,
                 h: int,
                 d_c: int = None,
                 d_o: int = None,
                 d_v: int = None,
                 optim: str = None,
                 head_split: bool = True,
                 dropout=0.1):
        """Multi-Headed Attention Layer

        Parameters
        ----------
        d_m: int
            Input model dimensions. Will be used for the dimensions of the final node embeddings.
        d_o: int
            Output dimension. This is particularly important when dealing with the decoder in the AM-D model. Here, the
            output is expected to be d_c, not d_m, which is different than in the encoder or the original Transformer.
        d_k: int
            Desired query-key dimensions. Used for the headed comparison.
        h: int
            Number of desired attention heads.
        d_c: int
            Dimensions of context. This is used exclusively by the decoder to specify a unique dimensionality of the
            context that is different than d_m. This is important, since the Attention Model does not apply a linear
            projection in the input like the Transformer.
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
        self.d_o = d_o if d_o is not None else d_m
        self.d_c = d_c
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
            # Self attention Optimization
            # First h * d_k outputs correspond to Q
            # Second h * d_k values correspond to K
            # Last h * d_v  values correspond to  V
            # [ Query | Key | Value ]
            '''For a more indepth explanation, see the else case bellow.
            To extract the Query, Key, and Value, we would need the following:
            1. Slice out Query, Key, and Value
            2. Reshape tensors to have dimensions: b x h x n_q/k x d_v,
            '''
            assert d_c is None, f"Cannot use a different d_m for query than key/value when applying self attention!"
            self.qkv_projection = nn.Linear(self.d_m, self.h * (self.d_q + self.d_k + self.d_v))
        elif optim == 'emb':
            # Non-self attention, using other embeddings.
            self.q_projection = nn.Linear(self.d_m if d_c is None else d_c, self.h * self.d_q)
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
            self.q_projection = nn.Linear(self.d_m if d_c is None else d_c, self.h * self.d_q) # d_q = d_k
            self.k_projection = nn.Linear(self.d_m, self.h * self.d_k)
            self.v_projection = nn.Linear(self.d_m, self.h * self.d_v)
        # Output linear projection from Vaswani paper.
        self.output  = nn.Linear(self.h * self.d_v, self.d_o)
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

        # QK -> batch x h x n_t x n_s
        QK = (Q @ K.transpose(-1, -2)) + (mask if mask is not None else 0)
        QK = QK / (self.d_k ** (1/2))
        # a -> batch x h x n_t x d_v

        a = self.softmax(QK.nan_to_num()).nan_to_num() @ V
        return a

    def forward(self,
                src: Tensor,
                emb: Tensor=None,
                mask: Tensor=None
                ) -> Tensor:
        if emb is None:
            emb = src

        '''Initial shapes
        src  -> b x n_q x  d_m or b x nodes x d_m for AM-D
        emb  -> b x n_kv x d_m or b x nodes x d_m for AM-D
        mask -> b x 1 x n_q x n_kv  or b x 1 x 1 x nodes for AM-D
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
            # K (reshape)-> b x n_q x h x d_k (transpose)-> b x h x n_q x d_k
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


class PersistenceEmbedding(nn.Module):
    def __init__(self,
                 d_m: int,
                 d_ff: int,
                 persistence_dimension: int = 2,
                 representation: str = "rips",
                 vectorization: str = "landscape",
                 options: dict = None,
                 max_edge_length: float=0.5):
        """Create Neural network that captures percistence information

        parameters
        ---------
        d_m: int
            Output dimension. input expected in shape batch x nodes x d_in. Output expected in
            shape batch x 1 x d_m
        d_ff: int
            Hidden layer of neural network.
                representation: str
        persistence_dimension: int
            Maximum dimension of persistence signatures to attend to.
        Topological representation to use. Options are:
            1. Rips Complex ('rips')
            2. Alpha Complex (Coming Soon!)
        vectorization: str
            Vectorization function into some hilbert space. Options are:
            1. Persistence Landscapes ('landscape')
            2. sillohuete (coming soon)
        options: dict
            Additional options for vectorization and representation functions.
            vectorization: 'landscape'
                resolution: int
                num_landscapes: int
        max_edge_length: float
            Maximum reach of filtration.
        """
        super().__init__()
        # Check that options are valid
        assert vectorization in ["landscape"], f"Invalid vectorization/representation {vectorization}."
        assert representation in ["rips"], f"Invalid representation {representation}. Only rips is supported."
        assert options is not None, f"Did not pass additional parameters!"

        # Set up topological representation of data.
        if representation == "rips":
            self.representation = gd.RipsComplex

        # Set up topological vectorization to hilbert-space
        if vectorization == "landscape":
            self.vectorization = gudhi.representations.Landscape(num_landscapes=options['num_landscapes'],
                                                              resolution=options['resolution'])
            self.num_landscapes = options['num_landscapes']
            self.resolution = options['resolution']
            self.d_vect = options['num_landscapes'] * options['resolution']

        # Additional parameters
        self.persistence_dimension = persistence_dimension
        # Input/output dimensions
        self.d_m = d_m
        # Neural net hidden layer
        self.d_ff = d_ff
        # Maximum reach of filtration.
        self.max_edge_length = max_edge_length
        # Create neural network
        self.layer_1 = nn.Linear(self.d_vect, d_ff)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(d_ff, d_m)

        # gd.representations.DiagramSelector(limit=np.inf, point_type="finite")

    def _get_persistence_vector(self,
                                 src: Tensor
                                 ) -> Tensor:
        """Compute persistence diagrams

        parameters
        ----------
        src: Tensor
            An input point cloud of shape nodes x d_m

        return
        ------
        Dg: Tensor
            Persistence diagram of shape per_points x 2, where per_points is highly dependent on the
            persistence signatures of the input.
        """
        # Most often, this is rips complex
        points = src.cpu().detach().numpy()
        cmpx = self.representation(points=points,
                                   max_edge_length=self.max_edge_length).create_simplex_tree(max_dimension=self.persistence_dimension)
        # Have filtration information available
        cmpx.compute_persistence()
        # Dg -> persistence features x 2 | Note: persistence features depends on input.
        Dg = cmpx.persistence_intervals_in_dimension(self.persistence_dimension - 1)
        # Dg_vect -> 1 x d_vect
        Dg_vect = self.vectorization.fit_transform([Dg])
        # Return -> d_vect
        return torch.FloatTensor(Dg_vect).squeeze()

    def forward(self,
                src: Tensor
                ) -> Tensor:
        """Calculate Persistence embedding

        Parameters
        -----------
        src: Tensor
            Input point cloud to be used for persistence computations. Expected
            shape: batch x nodes x d_m

        Returns
        -------
        persistence_embedding: Tensor
            The Persistence/structural information embedded into d_model
        """

        """Algorithm:
        compute persistance diagram
        Compute selected vectorization.
        pass vectorization through feed-forward neural network
        """
        device = list(self.parameters())[0].device

        # Batch-wise computation of hilbert space vector form of diagram.
        vectors = []
        for point_cloud in src:
            vect = self._get_persistence_vector(point_cloud)
            vectors.append(vect)
        # Dgms -> batch x d_vect
        Dgms = torch.stack(vectors).to(device)
        # ff_1 -> batch x d_ff
        ff_1 = self.relu(self.layer_1(Dgms))
        # ff_2 -> batch x d_m
        ff_2 = self.layer_2(ff_1)
        return ff_2
