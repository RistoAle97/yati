import torch
from torch import nn
from torch.functional import F

import yati.generation as inference
from yati.config import TransformerConfig
from yati.layers import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from yati.positional_encoding import PositionalEncoding
from yati.utils import init_bert_weights


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """
        The Transformer model from "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf).
        :param config: a TransformerConfig object.
        """
        super().__init__()
        # Parameters
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.num_encoder_layers = config.num_encoder_layers
        self.num_decoder_layers = config.num_decoder_layers
        self.dim_ff = config.dim_ff
        self.dropout = config.dropout
        self.dropout_mha = config.dropout_mha
        self.dropout_ff = config.dropout_ff
        self.activation_ff = config.activation_ff
        self.layer_norm_eps = config.layer_norm_eps

        # Token ids
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id

        # Embeddings and positional encoder
        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_token_id)
        self.positional_encoder = PositionalEncoding(self.d_model, dropout=self.dropout)
        self.embedding_scale = 1.0 if not config.scale_embeddings else self.d_model**0.5

        # Encoder
        encoder_norm = nn.LayerNorm(self.d_model, self.layer_norm_eps)
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dim_ff=self.dim_ff,
            dropout=self.dropout,
            dropout_mha=self.dropout_mha,
            dropout_ff=self.dropout_ff,
            activation_ff=self.activation_ff,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.encoder = TransformerEncoder(encoder_layer, self.num_encoder_layers, norm=encoder_norm)

        # Decoder
        decoder_norm = nn.LayerNorm(self.d_model, self.layer_norm_eps)
        decoder_layer = TransformerDecoderLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dim_ff=self.dim_ff,
            dropout=self.dropout,
            dropout_mha=self.dropout_mha,
            dropout_ff=self.dropout_ff,
            activation_ff=self.activation_ff,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.decoder = TransformerDecoder(decoder_layer, self.num_decoder_layers, norm=decoder_norm)

        # Linear output
        self.linear_output = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.tie_embeddings = config.tie_embeddings
        if self.tie_embeddings:
            self.linear_output.weight = self.embedding.weight

        # Label smoothing value
        self.label_smoothing = config.label_smoothing

        # Initialize weights
        self.apply(init_bert_weights)

    def encode(self, e_input: torch.Tensor, e_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encodes the masked source sentence.
        :param e_input: torch tensor of shape (bsz, seq_len).
        :param e_mask: mask for the encoder of shape (bsz, 1, seq_len).
        :return: torch tensor representing the encodings with shape (bsz, seq_len, d_model).
        """
        src_embeddings = self.embedding(e_input)  # (bsz, seq_len, d_model)
        src_embeddings = self.positional_encoder(src_embeddings * self.embedding_scale)
        e_output = self.encoder(src_embeddings, e_mask)
        return e_output

    def decode(
        self,
        tgt_input: torch.Tensor,
        e_output: torch.Tensor,
        d_mask: torch.Tensor = None,
        e_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Decodes the masked target sentence given the encodings of the source sentence.
        :param e_output: encodings coming from the encoder of shape (bsz, seq_len, d_model).
        :param tgt_input: torch tensor of shape (bsz, seq_len)
        :param e_mask: mask for the encoder of shape (bsz, 1, seq_len).
        :param d_mask: mask for the decoder of shape (bsz, seq_len, seq_len).
        :return: torch tensor representing the decodings with shape (bsz, seq_len, vocab_size).
        """
        tgt_embeddings = self.embedding(tgt_input)  # (bsz, seq_len, d_model)
        tgt_embeddings = self.positional_encoder(tgt_embeddings * self.embedding_scale)
        d_output = self.decoder(tgt_embeddings, e_output, d_mask, e_mask)
        return d_output

    def forward(
        self,
        src_input: torch.Tensor,
        tgt_input: torch.Tensor,
        e_mask: torch.Tensor = None,
        d_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Process masked source and target sequences.
        """
        # Embeddings and positional encoding
        src_embeddings = self.embedding(src_input)  # (bsz, seq_len, d_model)
        src_embeddings = self.positional_encoder(src_embeddings * self.embedding_scale)
        tgt_embeddings = self.embedding(tgt_input)  # (bsz, seq_len, d_model)
        tgt_embeddings = self.positional_encoder(tgt_embeddings * self.embedding_scale)

        # Encoder and decoder
        e_out = self.encoder(src_embeddings, e_mask)
        d_out = self.decoder(tgt_embeddings, e_out, d_mask, e_mask)

        # Linear output
        out = self.linear_output(d_out)  # (bsz, seq_len, vocab_size)
        return out

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)
        loss = F.cross_entropy(logits, labels, ignore_index=self.pad_token_id, label_smoothing=self.label_smoothing)
        return loss

    def generate(
        self,
        input_ids: torch.Tensor,
        decoder_start_token_id: int,
        max_new_tokens: int = 10,
        num_beams: int = 5,
    ) -> torch.Tensor:
        """
        Generate tokens at inference time using greedy or beam search decoding.
        :param input_ids: tokenized source sentence.
        :param decoder_start_token_id: the token that will prepend the output sequence, in a multilingual setting this
            should be the target language token, in a bilingual setting the beginning of sequence token should be
            used instead.
        :param max_new_tokens: the number of new tokens allowed on top of the source sentence length (default=10).
        :param num_beams: size of the beam, if it is equal to 1 than greedy decoding will be applied, otherwise
            beam search will be performed (default=5).
        :return: tokenized translation of the source sentence.
        """
        if num_beams < 1:
            raise ValueError("The beam size must be at least 1.")

        if max_new_tokens < 0:
            raise ValueError("The number of max new tokens must be at least 0.")

        self.eval()
        if num_beams == 1:
            output = inference.greedy_decoding(self, input_ids, decoder_start_token_id, max_new_tokens)
        else:
            output = inference.beam_decoding(self, input_ids, decoder_start_token_id, max_new_tokens, num_beams)

        return output
