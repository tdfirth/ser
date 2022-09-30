#created 29/09/22
from ser.CNN_model import Net
import torch

def inference(model_path):
	"Function to load and run a pretrained ML Model."
	
	print("Loading model...")
	model = Net()
	model.load_state_dict(torch.load(model_path))
	model.eval()

	return
