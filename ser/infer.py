import torch

from utils import generate_ascii_art


def infer_label(dataloader, path, label):
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))

    # load the model
    model = torch.load(path / "model.pt")

    # run inference
    model.eval()
    output = model(images)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    certainty = max(list(torch.exp(output)[0]))
    pixels = images[0][0]
    print(generate_ascii_art(pixels))
    print(f"This is a {pred}")
    print(f"Prediction confidence is {certainty}")
