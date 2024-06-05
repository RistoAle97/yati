from yati import SMALL_CONFIG, BASE_CONFIG, LARGE_CONFIG, Transformer, model_n_parameters
from yati.plots import plot_model_parameters

if __name__ == "__main__":
    model_small = Transformer(SMALL_CONFIG)
    print(model_n_parameters(model_small))
    plot_model_parameters(model_small)

    model_base = Transformer(BASE_CONFIG)
    print(model_n_parameters(model_base))
    plot_model_parameters(model_base)

    model_large = Transformer(LARGE_CONFIG)
    print(model_n_parameters(model_large))
    plot_model_parameters(model_large)
