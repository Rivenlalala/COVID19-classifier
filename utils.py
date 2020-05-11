from torchvision.datasets import ImageFolder
from torchvision import transforms


class CustomFolder(ImageFolder):
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform((sample, target))

        return sample
    

class CustomCompose(transforms.Compose):
    
    def __call__(self, sample):
        img = sample[0]
        label = sample[1]
        for t in self.transforms:
            if isinstance(t, CustomTransforms):
                img, label = t((img, label))
            else:
                img = t(img)
        return img, label
        

