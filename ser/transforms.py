from torchvision import transforms

# Torch transforms

def torchtransforms():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])