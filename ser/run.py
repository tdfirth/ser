import torch
from ser.select import select_image
from ser.artwork import generate_ascii_art

def run_inference(model, label):  
    # run inference
    model.eval()
    output = model(select_image(label)[1])
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    certainty = max(list(torch.exp(output)[0]))
    pixels = select_image(label)[1][0][0]
    print(generate_ascii_art(pixels))
    print(f"This is a {pred}. I am {int(certainty*100)}% confident that it is a {pred}.")

    return(output, pred, certainty, pixels)