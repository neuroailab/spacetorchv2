"""
Feature Extractor, inspired by Model Tools PytorchWrapper (DiCarlo lab)
"""

import math
from typing import Optional, List, Union, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from spacetorch.utils.torch_utils import resolve_sequential_module_from_str
from spacetorch.utils.generic_utils import make_deterministic


# ReturnedFeatures can be either
# 1. a single np array (single layer only)
# 2. a dictionary of string to np arrays (multiple layers)
# 3. a tuple of ({1, 2}, inputs, labels)
Features = Union[np.ndarray, Dict[str, np.ndarray]]
ReturnedFeatures = Union[Features, Tuple[Features, np.ndarray, np.ndarray]]


class FeatureExtractor:
    """
    Extracts activations from a layer of a model.

    Arguments:
        dataloader : (torch.utils.data.DataLoader) dataloader. assumes images
                     have been transformed correctly (i.e. ToTensor(),
                     Normalize(), Resize(), etc.)
        n_batches  : (int) number of batches to obtain image features
        vectorize  : (boolean) whether to convert layer features into vector
    """

    def __init__(
        self,
        dataloader: DataLoader,
        n_batches: Optional[int] = None,
        vectorize: bool = False,
        verbose: bool = True,
        spatial_mp: bool = False,
    ):
        self.dataloader = dataloader
        self.n_batches = n_batches or len(self.dataloader)
        self.vectorize = vectorize
        self.verbose = verbose
        self.spatial_mp = spatial_mp
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def extract_features(
        self,
        model: nn.Module,
        model_layer_strings: Union[str, List[str]],
        return_inputs_and_labels: bool = False,
        reshape: bool = True,
        reduce_along_hw: bool = False,
    ):

        if not isinstance(model_layer_strings, list):
            model_layer_strings = [model_layer_strings]

        assert len(model_layer_strings), "no layer strings provided"

        layer_results: Dict[str, Union[np.ndarray, List[np.ndarray]]] = {
            k: [] for k in model_layer_strings
        }
        hooks = []

        # add forward hooks to each model layer
        for layer_name in model_layer_strings:
            hook_layer_name = layer_name
            if "attn_matrix" in layer_name:
                hook_layer_name = hook_layer_name[:-12]
            if "attn_output" in layer_name:
                hook_layer_name = hook_layer_name[:-12]
            layer = resolve_sequential_module_from_str(model, hook_layer_name)
            hook = self.get_hook(layer, layer_name, target_dict=layer_results)
            hooks.append(hook)

        # switch model to eval mode
        model.eval()

        self.inputs = list()
        self.labels = list()

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(self.dataloader),
                total=self.n_batches,
                desc="batch",
                disable=not self.verbose,
            ):

                if batch_idx == self.n_batches:
                    break

                x, label_x = batch
                x = x.to(self.device)

                model(x)
                label_x = label_x.cpu().numpy()
                self.inputs.append(x.cpu().numpy())
                self.labels.append(label_x)

        self.labels = np.concatenate(self.labels)
        self.inputs = np.concatenate(self.inputs)

        # Reset forward hook so next time function runs, previous hooks
        # are removed
        for hook in hooks:
            hook.remove()

        self.layer_feats = {}
        for k, v in layer_results.items():
            self.layer_feats[k] = np.concatenate(v)
            
            if reshape:
                # to use queries, keys, or values from ViT blocks
                # self.layer_feats[k] = np.concatenate(v).reshape(640, 257, 3, 768).transpose(2, 0, 1, 3)[2]
                
                if len(self.layer_feats[k].shape) == 3:
                    # a hack to refer to outputs from vision transformers,
                    # removing the CLS token
                    N, L, H = self.layer_feats[k].shape

                    # Check if L is a perfect square
                    sL = int(math.sqrt(L))
                    is_square = sL * sL == L

                    # Remove CLS token only if L is not already a square
                    if not is_square:
                        self.layer_feats[k] = self.layer_feats[k][:, 1:]
                        L = self.layer_feats[k].shape[1]
                        sL = int(math.sqrt(L))

                    # Reshape to (N, C, H, W) format for compatibility with conv-style processing
                    self.layer_feats[k] = self.layer_feats[k].reshape(N, sL, sL, H).transpose(0, 3, 1, 2)

            if reduce_along_hw:
                # average pool along height and width dimensions
                self.layer_feats[k] = self.layer_feats[k].mean(axis=(-2, -1))

        # corner case: for a single layer, just return features
        if len(self.layer_feats) == 1:
            self.layer_feats = self.layer_feats[model_layer_strings[0]]

        if return_inputs_and_labels:
            return self.layer_feats, self.inputs, self.labels

        return self.layer_feats

    def get_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            if "attn_output" in name:
                output = _layer.attn_output
            if "attn_matrix" in name:
                output = _layer.attn_matrix

            if self.spatial_mp:
                output = torch.amax(output, dim=(2, 3))
            out = output.cpu().numpy()

            if self.vectorize:
                out = np.reshape(out, (len(out), -1))

            target_dict[name].append(out)

        hook = layer.register_forward_hook(hook_function)
        return hook


def get_features_from_layer(
    model: nn.Module,
    dataset: Dataset,
    model_layer_strings: Union[str, List[str]],
    batch_size: int = 32,
    max_batches: Optional[int] = None,
    return_inputs_and_labels: bool = False,
    shuffle: bool = True,
    verbose: bool = True,
    spatial_mp: bool = False,
    reshape: bool = True,
    reduce_along_hw: bool = False,
    num_workers: int = 1,
) -> ReturnedFeatures:
    """Workhorse feature extractor.

    Inputs:
        model: the pytorch module to run images through
        dataset: dataset to make a dataloader out of, to get image samples
        model_layer_strings: either a single string, or a list of strings indicating
            which layers to get features from.
        batch_size: self-explanatory
        max_batches: either None (iterate the whole dataset), or the maximum number
            of batches to use
        return_inputs_and_labels: if True return a tuple with (Features, inputs, labels)
        shuffle: if True, shuffle the dataset before use
        verbose: whether to print progress with tqdm
        num_workers: dataloader worker processes. The default of 1 is enough
            for the small local stimulus sets, but reading ImageNet off network
            storage is decode-bound and wants considerably more.

    Returns:
        ReturnedFeatures: either a tuple including features, inputs, and labels, or just
            features. Features can be either a single numpy array, or a dictionary
            mapping from layer names to numpy arrays.
    """

    n_images: int = len(dataset)  # type: ignore
    n_batches: int = max_batches or math.ceil(n_images / batch_size)

    make_deterministic(seed=1)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    extractor = FeatureExtractor(
        dataloader, n_batches, verbose=verbose, spatial_mp=spatial_mp
    )
    return extractor.extract_features(
        model, model_layer_strings, return_inputs_and_labels=return_inputs_and_labels, reshape=reshape, reduce_along_hw=reduce_along_hw
    )


def get_pooled_features_from_layer(
    model: nn.Module,
    dataset: Dataset,
    model_layer_strings: Union[str, List[str]],
    reducer,
    indices: Optional[np.ndarray] = None,
    batch_size: int = 32,
    chunk_size: int = 512,
    num_workers: int = 1,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Extract features in chunks, reducing each chunk before the next is read.

    ``get_features_from_layer`` holds every image's full feature map in memory
    at once. That is fine for a late layer, but a single early convolutional
    layer over a few thousand images runs to several gigabytes, and the orbit
    analyses need the same stimuli through many models. This variant keeps only
    ``chunk_size`` images of feature map alive at a time and stores the reduced
    result, which for the privileged-axis analyses is a handful of numbers per
    channel per image.

    Several layers can be requested at once, and should be whenever more than
    one is wanted from the same images: reading and decoding the stimuli
    dominates the cost, so hooking four layers in one pass is close to four
    times faster than four passes.

    Inputs:
        reducer: callable mapping (features, labels) for one chunk of one layer
            to an array whose first dimension is the chunk's images
        indices: optional subset of the dataset, used in the given order; the
            same indices through two models therefore yield row-aligned
            responses

    Returns:
        ({layer: reduced}, labels), concatenated over chunks in dataset order
    """
    from torch.utils.data import Subset

    if not isinstance(model_layer_strings, list):
        model_layer_strings = [model_layer_strings]

    if indices is None:
        indices = np.arange(len(dataset))  # type: ignore
    indices = np.asarray(indices)

    reduced_chunks: Dict[str, List[np.ndarray]] = {k: [] for k in model_layer_strings}
    label_chunks = []
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start : start + chunk_size]
        if verbose:
            print(f"images {start}-{start + len(chunk)} of {len(indices)}", flush=True)
        features, _, labels = get_features_from_layer(
            model,
            Subset(dataset, chunk),
            model_layer_strings=model_layer_strings,
            batch_size=batch_size,
            return_inputs_and_labels=True,
            shuffle=False,
            verbose=False,
            num_workers=num_workers,
        )
        if not isinstance(features, dict):
            features = {model_layer_strings[0]: features}
        for layer_name, layer_features in features.items():
            reduced_chunks[layer_name].append(reducer(layer_features, labels))
        label_chunks.append(labels)
        del features

    reduced = {k: np.concatenate(v) for k, v in reduced_chunks.items()}
    return reduced, np.concatenate(label_chunks)
