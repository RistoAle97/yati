import matplotlib.pyplot as plt

from yati.transformer import Transformer
from yati.utils import model_n_parameters


def plot_model_parameters(model: Transformer, only_trainable: bool = False) -> None:
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
    from numerize.numerize import numerize
    labels = [label + f"\n{numerize(value)}" for label, value in zip(labels, values)]

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
    plt.title(f"Model parameters ({numerize(model_parameters)}) distribution", fontsize=18, weight="bold", pad=15)
    fig.show()
