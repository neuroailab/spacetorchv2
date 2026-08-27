import torchvision


DISCRIMINATION_TRANSFORMS = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ]
)


NUM_TRAIN_IMAGES = 6400
NUM_VALIDATION_IMAGES = 6400


class DiscriminationData(torchvision.datasets.ImageFolder):
    """Same/Different Task (Lepori et al., 2024)"""

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
