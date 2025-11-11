import six
import torch
from PIL import Image
from torchvision import models
from torch_hub import TorchHub

class Utils:
    def setup_test():
        x = torch.rand(5, 3)
        print(x)

        if torch.cuda.is_available():
            print("CUDA is available!")

    def show_available_models():
        print(dir(models))

    def load_pretrained_weights(model_path: str, weights_only=True):        
        if (not model_path):
            print("model_path is None, (invalid input)")
            return

        return torch.load(model_path, weights_only= weights_only)        

    def get_image(file_name: str, mustShow = False) -> Image.ImageFile:
        if (not file_name):
            print("file_name is None, (invalid input)")
            return

        img = Image.open(file_name)
        print(img)  

        if img.mode != 'RGB':
            img = img.convert('RGB')

        if mustShow: img.show()
        return img
    
    def compare_models(model_data, model: torch.nn.Module):
        model_keys = model_data.keys() 
        netG_keys = model.state_dict().keys()
        assert model_keys == netG_keys, "model data keys and model keys are different."

    def get_device() -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'    
    
    def load_pickle(file_path: str):    
        if (not file_path):
            print("file_path is None, (invalid input)")
            return
        
        try:
            with open(file_path, 'rb') as f:
                return six.moves.cPickle.load(f, encoding='latin1')                
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")

    def get_model_from_hub(hub: TorchHub):
        if (not hub):
            print("hub is None, (invalid input)")
            return
        
        return hub.get_model()