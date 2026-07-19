# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

from torchvision import datasets


class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        image = self.loader(path)
        
        ret = []
        if self.transform is not None:
            try:
                for t in self.transform:
                    ret.append(t(image))
            except Exception as e:
                ret.append(self.transform(image))
        else:
            ret.append(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        ret.append(target)

        return ret