"""Configuration class for all the Transformers."""


class TransformerConfig:
    """Configuration class for the Transformer model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        dropout_mha: float = 0.0,
        dropout_ff: float = 0.0,
        activation_ff: str = "relu",
        layer_norm_eps: float = 1e-6,
        scale_embeddings: bool = False,
        tie_embeddings: bool = True,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        pad_token_id: int = 1,
        label_smoothing: float = 0.0,
    ) -> None:
        """Initialize a TransformerConfig object.

        Args:
            vocab_size: the shared vocabulary size.
            d_model: embedding dimension (default=512).
            n_heads: the number of heads in the multi-attention mechanism (default=8).
            num_encoder_layers: the number of encoder layers (default=6).
            num_decoder_layers: the number of decoder layers (default=6).
            dim_ff: dimension of the feedforward sublayer (default=2048).
            dropout: the dropout value (default=0.1).
            dropout_mha: the dropout value for the multi-head attention (default=0.0).
            dropout_ff: the dropout value for the feed-forward sublayer (default=0.0).
            activation_ff: the activation function for the feed-forward sub-layer, can be either ReLU or GeLU
                (default="relu").
            layer_norm_eps: the eps value in the layer normalization (default=1e-6).
            scale_embeddings: whether to scale the output of the embedding layer with the inverse square root of
                ``d_model`` (default=False).
            tie_embeddings: whether to tie the decoder embeddings to the linear output layer (default=True).
            bos_token_id: the beginning of sequence token id (default=0).
            eos_token_id: the end of sequence token id (default=2).
            pad_token_id: the pad token id (default=1).
            label_smoothing: the label smoothing value for the cross-entropy loss (default=0.0).
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.dropout_mha = dropout_mha
        self.dropout_ff = dropout_ff
        self.activation_ff = activation_ff
        self.layer_norm_eps = layer_norm_eps
        self.scale_embeddings = scale_embeddings
        self.tie_embeddings = tie_embeddings
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing


# Some predefined configurations
SMALL_CONFIG = TransformerConfig(32000, d_model=256)  # 25.5m parameters
BASE_CONFIG = TransformerConfig(32000)  # 60.5m parameters
LARGE_CONFIG = TransformerConfig(32000, d_model=1024, n_heads=16, dim_ff=4096, dropout=0.3)  # 209m parameters
