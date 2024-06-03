import copy

import torch
from torch import nn
from torch.functional import F


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.0) -> None:
        """
        The multi-head attention sub-layer from "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf).
        :param d_model: the model's embedding dimension (default=512).
        :param n_heads: the number of heads (default=8).
        :param dropout: the dropout value (default= 0.0).
        """
        super().__init__()
        assert d_model % n_heads == 0

        # Parameters
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Linear projections for query, key and value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection and dropout
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def __self_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        attn_scores /= self.d_model**0.5

        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(torch.as_tensor(mask == 0), float("-inf"))

        p_attn = attn_scores.softmax(dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value)  # (bsz, n_heads, seq_len, d_model)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Get the batch size
        bsz = query.size(0)

        # Perform linear operation and split into n_heads heads
        k = self.k_proj(key).view(bsz, -1, self.n_heads, self.head_dim)  # (bsz, seq_len, n_heads, head_dim)
        q = self.q_proj(query).view(bsz, -1, self.n_heads, self.head_dim)  # (bsz, seq_len, n_heads, head_dim)
        v = self.v_proj(value).view(bsz, -1, self.n_heads, self.head_dim)  # (bsz, seq_len, n_heads, head_dim)

        # Transpose to get shapes (bsz, n_heads, seq_len, d_model)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute self-attention
        scores = self.__self_attention(q, k, v, mask)  # (bsz, n_heads, seq_len, d_model)

        # Concatenate heads and pass through the final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bsz, -1, self.d_model)
        output = self.out_proj(concat)  # (bsz, seq_len, d_model)
        return output


class FeedForward(nn.Module):

    def __init__(self, d_model: int = 512, dim_ff: int = 2048, dropout: float = 0.0, activation: str = "relu") -> None:
        """
        The feed-forward sub-layer from "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf).
        :param d_model: the model's embedding dimension (default=512).
        :param dim_ff: size of the intermediate linear transformation (default=2048).
        :param dropout: the dropout value (default=0.0).
        :param activation: the activation function, can be either ReLU or GeLu (default="relu").
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, d_model)
        if activation not in ["relu", "gelu"]:
            raise ValueError(
                "The activation function of the feed-forward sub-layer must be either \"relu\" or \"gelu\"."
            )

        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear1(x))  # (bsz, seq_len, dim_ff)
        out = self.dropout(out)
        out = self.linear2(out)  # (bsz, seq_len, d_model)
        return out


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        dropout_mha: float = 0.0,
        dropout_ff: float = 0.0,
        activation_ff: str = "relu",
        layer_norm_eps: float = 1e-6,
    ) -> None:
        """
        The transformer encoder layer from "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf). The layer
        is made up of one multi-head attention sub-layer followed by a feed-forward sub-layer. Differently from the
        paper, this implementation uses pre-norm inside the residual connection.
        :param d_model: the model's embedding dimension (default=512).
        :param n_heads: the number of heads in the multi-attention mechanism (default=8).
        :param dim_ff: dimension of the feedforward sub-layer (default=2048).
        :param dropout: the dropout value used at the end of each sub-layer (default=0.1).
        :param dropout_mha: the dropout value for the multi-head attention (default=0.0).
        :param dropout_ff: the dropout value for the feed-forward sub-layer (default=0.0).
        :param activation_ff: the activation function for the feed-forward sub-layer, can be either ReLU or GeLU
            (default="relu").
        :param layer_norm_eps: the eps value in the layer normalization (default=1e-6).
        """
        super().__init__()
        # Multi-head attention sub-layer
        self.mha_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout_mha)
        self.mha_dropout = nn.Dropout(dropout)

        # Feed-forward sub-layer
        self.ff_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.ff = FeedForward(d_model, dim_ff, dropout_ff, activation_ff)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, src_embeddings: torch.Tensor, e_mask: torch.Tensor = None) -> torch.Tensor:
        # Multi-head attention sub-layer
        mha_out = self.mha_norm(src_embeddings)
        mha_out = self.mha(mha_out, mha_out, mha_out, e_mask)
        mha_out = src_embeddings + self.mha_dropout(mha_out)

        # Feed-forward sub-layer
        ff_out = self.ff_norm(mha_out)
        ff_out = self.ff(ff_out)
        out = mha_out + self.ff_dropout(ff_out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, e_layer: TransformerEncoderLayer, num_layers: int = 6, norm: nn.LayerNorm = None) -> None:
        """
        The encoder from "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf). Following the actual
        implementation of the paper, a LayerNorm layer is put at the end of the encoder layers stack.
        :param e_layer: transformer's encoder layer that will be used in order to build the stack of encoder layers.
        :param num_layers: the number of layers (default=6).
        :param norm: the layer normalization that should be at the end of the encoder layers stack (default=None).
        """
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(e_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src_embeddings: torch.Tensor, e_mask: torch.Tensor = None) -> torch.Tensor:
        e_out = src_embeddings
        for encoder_layer in self.layers:
            e_out = encoder_layer(e_out, e_mask)

        if self.norm is not None:
            e_out = self.norm(e_out)

        return e_out


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        dropout_mha: float = 0.0,
        dropout_ff: float = 0.0,
        activation_ff: str = "relu",
        layer_norm_eps: float = 1e-6,
    ) -> None:
        """
        The transformer decoder layer from "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf). The layer
        is made up of two multi-head attention sub-layers (self-attention and encoder-decoder cross-attention) followed
        by a feed-forward sub-layer. Differently from the paper, this implementation uses pre-norm inside the residual
        connection.
        :param d_model: the model's embedding dimension (default=512).
        :param n_heads: the number of heads in the multi-attention mechanism (default=8).
        :param dim_ff: dimension of the feedforward sub-layer (default=2048).
        :param dropout: the dropout value used at the end of each sub-layer (default=0.1).
        :param dropout_mha: the dropout value for the multi-head attention (default=0.0).
        :param dropout_ff: the dropout value for the feed-forward sub-layer (default=0.0).
        :param activation_ff: the activation function for the feed-forward sub-layer, can be either ReLU or GeLU
            (default="relu").
        :param layer_norm_eps: the eps value in the layer normalization (default=1e-6).
        """
        super().__init__()
        # Multi-head attention sub-layer
        self.mha_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout_mha)
        self.mha_dropout = nn.Dropout(dropout)

        # Cross-attention sub-layer
        self.ca_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.ca = MultiHeadAttention(d_model, n_heads, dropout_mha)
        self.ca_dropout = nn.Dropout(dropout)

        # Feed-forward sub-layer
        self.ff_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.ff = FeedForward(d_model, dim_ff, dropout_ff, activation_ff)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt_embeddings: torch.Tensor,
        e_output: torch.Tensor,
        d_mask: torch.Tensor = None,
        e_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Multi-head attention sub-layer
        mha_out = self.mha_norm(tgt_embeddings)
        mha_out = self.mha(mha_out, mha_out, mha_out, d_mask)
        mha_out = tgt_embeddings + self.mha_dropout(mha_out)

        # Cross-attention sub-layer
        ca_out = self.ca_norm(mha_out)
        ca_out = self.ca(ca_out, e_output, e_output, e_mask)
        ca_out = mha_out + self.mha_dropout(ca_out)

        # Feed-forward sub-layer
        ff_out = self.ff_norm(ca_out)
        ff_out = self.ff(ff_out)
        out = ca_out + self.ff_dropout(ff_out)
        return out


class TransformerDecoder(nn.Module):
    def __init__(self, d_layer: TransformerDecoderLayer, num_layers: int = 6, norm: nn.LayerNorm = None):
        """
        The decoder from "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf). Following the actual
        implementation of the paper, a LayerNorm layer is put at the end of the decoder layers stack.
        :param d_layer: transformer's decoder layer that will be used in order to build the stack of decoder layers.
        :param num_layers: the number of layers (default=6).
        :param norm: the layer normalization that should be at the end of the decoder layers stack (default=None).
        """
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(d_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(
        self,
        tgt_embeddings: torch.Tensor,
        e_out: torch.Tensor,
        d_mask: torch.Tensor = None,
        e_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        d_out = tgt_embeddings
        for decoder_layer in self.layers:
            d_out = decoder_layer(d_out, e_out, d_mask, e_mask)

        if self.norm is not None:
            d_out = self.norm(d_out)

        return d_out
