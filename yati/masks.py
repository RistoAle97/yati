import torch


def generate_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    :param seq_len: length of the sequence to mask.
    :return: causal mask for the autoregressive decoder.
    """
    return torch.tril(torch.ones(1, seq_len, seq_len, dtype=torch.bool))  # (1, seq_len, seq_len)


def create_encoder_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Create the mask for the encoder of a transformer-based model, this only considers the pad tokens inside the
    input ids.
    :param input_ids: the encoder's input tokens.
    :param pad_token_id: the pad token id.
    :return: the mask for the encoder of shape (bsz, 1, seq_len).
    """
    return input_ids.ne(pad_token_id).unsqueeze(1).to(input_ids.device)  # (bsz, 1, seq_len)


def create_decoder_mask(decoder_input_ids: torch.Tensor, pad_token_id: int, is_causal: bool = True) -> torch.Tensor:
    """
    Create the mask for the decoder of a transformer-based model, the mask is formed by combining the pad mask to a
    decoder mask specified by the user.
    :param decoder_input_ids: the decoder's input tokens.
    :param pad_token_id: the pad token id.
    :param is_causal: whether the decoder mask is causal, if False only the padding mask for the decoder will be built.
       (default=True).
    :return: the mask for the decoder of shape (bsz, seq_len, seq_len).
    """
    d_pad_mask = decoder_input_ids.ne(pad_token_id).unsqueeze(1).to(decoder_input_ids.device)
    seq_len = decoder_input_ids.size(-1)
    if is_causal:
        nopeak_mask = generate_causal_mask(seq_len).to(decoder_input_ids.device)
    else:
        nopeak_mask = torch.ones(1, seq_len, seq_len, dtype=torch.bool, device=decoder_input_ids.device)

    d_mask = d_pad_mask & nopeak_mask
    return d_mask  # (bsz, seq_len, seq_len)


def create_masks(
    input_ids: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    pad_token_id: int,
    is_causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create masks for both encoder and decoder. The encoder's mask will prevent the module from attending on padding
    tokens, while the decoder's mask will also prevent the module from attending on user-specified tokens
    (e.g.: prevent the model to look ahead by using a causal mask).
    :param input_ids: the encoder's input tokens.
    :param decoder_input_ids: the decoder's input tokens.
    :param pad_token_id: the model pad token id.
    :param is_causal: whether the decoder mask is causal, if False only the padding mask for the decoder will be built.
       (default=True).
    :return: the masks for both the encoder and decoder of a transformer-based model.
    """
    e_mask = create_encoder_mask(input_ids, pad_token_id)  # (bsz, 1, seq_len)
    d_mask = create_decoder_mask(decoder_input_ids, pad_token_id, is_causal)  # (bsz, seq_len, seq_len)
    return e_mask, d_mask
