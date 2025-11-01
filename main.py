from utils import Utils
from image_generation import ImageGeneration
from image_captioning import ImageCaptioning
from image_classification import ImageClassification

def run_image_classification():
    image = Utils.get_image("imgs/dog.jpg")
    out_tensor = ImageClassification.start_pipeline(image)
    ImageClassification.predict(out_tensor, topPredicts=5)

def run_image_generation():
    image = Utils.get_image("imgs/horse.png")
    batch_out = ImageGeneration.start_pipeline(image)
    ImageGeneration.normalize_img(batch_out, mustShow=True, mustSave=True)

def run_image_captioning():
    image = Utils.get_image("imgs/adventure_life.png")
    ImageCaptioning.predict(image)

# run_image_classification()
# run_image_generation()
run_image_captioning()