import torch

from ser.data import get_data
from ser.transforms import normalize, flip, transforms
from utils import generate_ascii_art


@torch.no_grad()
def infer_label(model, image):
    # Infer label and calculate certainty

    model.eval()
    output = model(image)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    certainty = max(list(torch.exp(output)[0]))

    # Generate the ascii art to see what the image looks like
    pixels = image[0][0]
    print(generate_ascii_art(pixels))

    print(f"I am {certainty * 100:.2f}% certain that it's a... {pred}\n")


def select_test_image(label, transform_list):
    dataloader = get_data(
        transform=transforms(*transform_list), batch_size=1, train=False
    )
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))
    return images
