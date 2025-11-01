import torch
from PIL import Image
from torchvision import models
from torchvision import transforms

class ImageClassification:

    def __get_alexNet() -> models.AlexNet:
        alexNet = models.AlexNet()
        print(alexNet)
        return alexNet
            
    def __get_resNet() -> models.ResNet:
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        print(resnet)
        return resnet    

    def __get_transform_compose() -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
        )])

    def __load_labels() -> list[str]:
        with open("utils_files/imagenet_classes.txt") as f:
            return [line.strip() for line in f.readlines()]       

    def start_pipeline(img: Image.ImageFile):
        
        if (not img):
            print("transform or img is None, (invalid inputs)")
            return
        
        restnet = ImageClassification.__get_resNet()    
        restnet.eval()

        transform = ImageClassification.__get_transform_compose()
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        
        out = restnet(batch_t)
        print(out)
        return out

    def predict(out_tensor):
        _, index = torch.max(out_tensor, 1)
        percentage = torch.nn.functional.softmax(out_tensor, dim=1)[0] * 100
        labels = ImageClassification.__load_labels()

        print("prediction is: ", labels[index[0]])
        print("accuracy of: ", percentage[index[0]].item())

    def predict(out_tensor, topPredicts = 5):   
        _, indices = torch.sort(out_tensor, descending=True)
        percentage = torch.nn.functional.softmax(out_tensor, dim=1)[0] * 100    
        labels = ImageClassification.__load_labels()

        [
            (print("prediction is: ", labels[idx]), 
            print("accuracy of: ", percentage[idx].item())) 
            for idx in indices[0][:topPredicts]
        ]