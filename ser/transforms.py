from torchvision import transforms

# Torch transforms

def torchtransforms():
    transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])