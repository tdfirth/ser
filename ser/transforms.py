#all data transforms defined here
from torchvision import transforms

# torch transforms
def transform(type: str = 'basic'):
    if type == 'basic':
        ts = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
    elif str.lower(type)=='train':
       ts = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            #add some augmentations
        ) 
    return ts
