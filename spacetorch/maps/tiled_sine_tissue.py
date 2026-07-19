from spacetorch.datasets import DatasetRegistry
from spacetorch.feature_extractor import get_features_from_layer
from spacetorch.datasets.tiled_sine_gratings import TiledSineResponses


def get_tiled_sine_responses(
    model,
    layers,
    architecture: str,
    verbose: bool = True,
    normalize_to_ringach_firing_rates: bool = True,
    reduce_along_hw: bool = False,
    is_llcnn: bool = False,
    is_lcnn: bool = False,
) -> TiledSineResponses:
    dataset = DatasetRegistry.get("TiledSineGrating2019")
    sine_features, _, sine_labels = get_features_from_layer(
        model,
        dataset,
        model_layer_strings=layers,
        verbose=verbose,
        return_inputs_and_labels=True,
        reduce_along_hw=reduce_along_hw,
    )

    return TiledSineResponses(sine_features, sine_labels, grid_shape=dataset.grid_shape, architecture=architecture, layer=layers[0], normalize_to_ringach_firing_rates=normalize_to_ringach_firing_rates, is_llcnn=is_llcnn, is_lcnn=is_lcnn)
