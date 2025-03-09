> 🚨 **WARNING: This package is still under development and NOT ready for production use!** 🚨

# AugmentDataset

`AugmentDataset` is a Python class that extends a dataset by adding augmented versions of selected samples. It allows you to apply random data augmentations on a dataset with control over the number of transformations, the probability of augmentation, and the number of augmented samples generated from each original sample.

## Installation

```bash
pip install git+https://github.com/raideno/data-augmentation.git
```

## Usage

### Example

```python
from data_augmentation import AugmentDataset

# Define your dataset (e.g., a PyTorch Dataset or a custom dataset)
class MyDataset:
    def __getitem__(self, index):
        # Return your data sample here
        pass

    def __len__(self):
        # Return the length of your dataset
        pass

# Create your dataset object
dataset = MyDataset()

# Define your augmentations (e.g., simple functions like flipping, rotation, etc.)
def rotate(sample):
    # Rotate the sample by 90 degrees
    return sample

def flip(sample):
    # Flip the sample horizontally
    return sample

transforms = [rotate, flip]

# Define your AugmentDataset
augment_dataset = AugmentDataset(
    dataset=dataset,
    probability_to_augment=0.5,    # 50% chance of augmentation per sample
    transforms=transforms,         # List of transformations to apply
    probabilities=[0.7, 0.3],      # Probabilities for each transformation
    max_transforms_per_sample=2,   # Apply a maximum of 2 transformations per sample
    augmentations_per_sample=3     # Create 3 augmented versions per sample
)

# Access original or augmented samples
sample = augment_dataset[0]  # Get the first sample (original or augmented)
```
