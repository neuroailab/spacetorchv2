from spacetorch.datasets import DatasetRegistry
from spacetorch.feature_extractor import get_features_from_layer
from spacetorch.datasets.expanding_sine_gratings import ExpandingSineResponses


def get_expanding_sine_responses(
    model,
    layers,
    verbose: bool = True,
    normalize_to_ringach_firing_rates: bool = True,
    reduce_along_hw: bool = False,
    is_llcnn: bool = False,
    is_lcnn: bool = False,
    contrast: str = "high",
) -> ExpandingSineResponses:
    dataset = DatasetRegistry.get("ExpandingSineGrating2019" if contrast == "high" else "ExpandingSineGratingLC2026")
    sine_features, _, sine_labels = get_features_from_layer(
        model,
        dataset,
        model_layer_strings=layers,
        verbose=verbose,
        return_inputs_and_labels=True,
        reduce_along_hw=reduce_along_hw,
    )

    return ExpandingSineResponses(sine_features, sine_labels, normalize_to_ringach_firing_rates=normalize_to_ringach_firing_rates, is_llcnn=is_llcnn, is_lcnn=is_lcnn)
