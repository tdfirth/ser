import torch


@torch.no_grad()
def infer(params, model, image, label):
    print(f"\nRunning inference for the model\n -  {params.name}")
    print(f"It was trained with the following hyperparameters:")
    print(f"  - Epochs:        {params.epochs}")
    print(f"  - Batch Size:    {params.batch_size}")
    print(f"  - Learning Rate: {params.learning_rate}\n")
    print(f"The image you have asked to classify is a {label}.")

    # Infer label and calculate certainty
    model.eval()
    output = model(image)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    certainty = max(list(torch.exp(output)[0]))

    # Generate the ascii art to see what the image looks like
    pixels = image[0][0]
    print(generate_ascii_art(pixels))

    print(f"I am {certainty * 100:.2f}% certain that it's a... {pred}\n")


def generate_ascii_art(pixels):
    ascii_art = []
    for row in pixels:
        line = []
        for pixel in row:
            line.append(pixel_to_char(pixel))
        ascii_art.append("".join(line))
    return "\n".join(ascii_art)


def pixel_to_char(pixel):
    if pixel > 0.99:
        return "O"
    elif pixel > 0.9:
        return "o"
    elif pixel > 0:
        return "."
    else:
        return " "
