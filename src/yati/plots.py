"""Utils for plotting a graph about the distribution of parameters inside a transformer model."""

import matplotlib.pyplot as plt

from .transformer import Transformer
from .utils import model_n_parameters


def _numerize(n: int, ndigits: int = 2) -> str:
    if n < 0:
        raise ValueError("The value must be positive")

    conversion_tuples = [(1e0, ""), (1e3, "K"), (1e6, "M"), (1e9, "B"), (1e12, "T")]
    for i, conv_tuple in enumerate(conversion_tuples):
        key, value = conv_tuple
        if n >= key:
            continue
        else:
            prev_key, prev_value = conversion_tuples[i - 1]
            return str(round(n / prev_key, ndigits)) + prev_value

    last_key, last_value = conversion_tuples[-1]
    return str(round(n / last_key, ndigits)) + last_value


def plot_model_parameters(model: Transformer, only_trainable: bool = False) -> None:
    """Plots the distribution of the model's parameters.

    Args:
        model: the model whose parameters should be plotted.
        only_trainable: whether to consider only the trainable parameters.
    """
    # Compute the number of parameters for each module
    idx_to_consider = 1 if only_trainable else 0
    embeddings_parameters = model_n_parameters(model.embedding)[idx_to_consider]
    e_parameters = model_n_parameters(model.encoder)[idx_to_consider]
    d_parameters = model_n_parameters(model.decoder)[idx_to_consider]

    # Put the labels and the values inside two lists
    labels = ["Embeddings", "Encoder", "Decoder"]
    values = [embeddings_parameters, e_parameters, d_parameters]

    # Take care of the possibility of tied embeddings with the linear output
    if not model.tie_embeddings:
        lin_out_parameters = model_n_parameters(model.linear_output)[idx_to_consider]
        labels.append("Linear Output")
        values.append(lin_out_parameters)

    # Compute the total number of parameters
    model_parameters = sum(values)
    labels = [label + f"\n{_numerize(value)}" for label, value in zip(labels, values)]

    # Build the plot
    fig, ax = plt.subplots(figsize=(7, 5))
    wedgeprops = {"edgecolor": "k", "linewidth": 2, "antialiased": True}
    _, labels_plot, autopcts = ax.pie(
        values,
        labels=labels,
        autopct="%1.2f%%",
        wedgeprops=wedgeprops,
        labeldistance=1.2,
    )
    plt.setp(labels_plot, **{"weight": "bold", "fontsize": 15.5})
    plt.setp(autopcts, **{"color": "white", "weight": "bold", "fontsize": 14.5})
    plt.title(f"Model parameters ({_numerize(model_parameters)}) distribution", fontsize=18, weight="bold", pad=15)
    fig.show()
