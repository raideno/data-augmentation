import random

from typing import List, Callable

class AugmentDataset():
    """
    A dataset wrapper that applies random data augmentations to specified samples.

    This class extends an existing dataset by adding augmented versions of selected samples.
    It allows control over the probability of augmentation, the maximum number of transformations
    per sample, and the number of augmented samples to generate for each original sample.

    Args:
        dataset (Dataset): The original dataset that implements the __getitem__ and __len__ methods.
        probability_to_augment (float): Probability of applying a transformation to a given sample. 
                                        Must be in the range [0, 1].
        transforms (List[Callable]): A list of transformation functions to apply for data augmentation.
        probabilities (List[float], optional): A list of probabilities corresponding to each transform in 
                                               `transforms`. If None, all transforms are assigned equal probability.
        max_transforms_per_sample (int): Maximum number of transformations to apply to each sample. Default is 1.
        augmentations_per_sample (int): Number of augmented samples to generate from one sample. Default is 1.

    Attributes:
        dataset (Dataset): The original dataset.
        transforms (List[Callable]): The list of transformation functions.
        probabilities (List[float]): Probabilities for each transformation.
        samples_to_augment (List[int]): List of sample indices chosen for augmentation.
        max_transforms_per_sample (int): Maximum number of transformations applied to each sample.
        augmentations_per_sample (int): Number of augmented versions created for each sample.

    Methods:
        __len__(): Returns the total number of samples, including augmented ones.
        __getitem__(index): Retrieves the sample at the given index, applying augmentations if applicable.
    """
    def __init__(
        self,
        dataset,
        probability_to_augment: float,
        transforms: List[Callable],
        probabilities: List[float] = None,
        max_transforms_per_sample: int = 1,
        augmentations_per_sample: int = 1,
    ):
        self.dataset = dataset
        self.probability_to_augment = probability_to_augment
        self.transforms = transforms
        self.probabilities = probabilities if probabilities is not None else [1/len(transforms)] * len(transforms)
        self.max_transforms_per_sample = max_transforms_per_sample
        self.augmentations_per_sample = augmentations_per_sample
        
        self.__validate_parameters()
        
        # NOTE: each selected sample will contribute at most `augmentations_per_sample` entries to the new augmented dataset, it can contribute less or nothing
        self.samples_to_augment = [
            i for i in range(len(dataset)) 
            for _ in range(augmentations_per_sample)
            if random.random() < probability_to_augment
        ]
        
    def __validate_parameters(self):
        if not hasattr(self.dataset, '__getitem__') or not hasattr(self.dataset, '__len__'):
            raise TypeError("The provided dataset must implement both `__getitem__` and `__len__` methods.")
     
        if not all(callable(transform) for transform in self.transforms):
            raise TypeError("All elements in the `transforms` list must be callable functions.")
     
        if len(self.transforms) != len(self.probabilities):
            raise ValueError("The number of transformations must be equal to the number of probabilities.")
        
        if not all([0 <= prob <= 1 for prob in self.probabilities]):
            raise ValueError("Probabilities must be in the range [0, 1].")
        
        if self.max_transforms_per_sample < 1:
            raise ValueError("The maximum number of transformations per sample must be at least 1.")
        
        if self.augmentations_per_sample < 1:
            raise ValueError("The number of augmentations per sample must be at least 1.")
        
    def __len__(self):
        return len(self.dataset) + len(self.samples_to_augment)
        
    def __getitem__(self, index):
        if index < len(self.dataset):
            return self.dataset[index]
        
        augmented_sample_index = index - len(self.dataset)
        sample_index = self.samples_to_augment[augmented_sample_index]
        augmented_sample = self.dataset[sample_index]

        transforms_to_apply = random.choices(a=self.transforms, p=self.probabilities, k=random.randint(1, self.max_transforms_per_sample))

        for transform in transforms_to_apply:
            augmented_sample = transform(augmented_sample)

        return augmented_sample