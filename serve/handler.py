# https://github.com/pytorch/serve/tree/master/model-archiver
from cProfile import label
import torch
import torch.nn.functional as F
from torchvision import transforms
import subprocess
from pathlib import Path

import fitz
import einops
from PIL import Image

from ts.torch_handler import base_handler
import ts
#import symposium
#import parameters
#import train

BATCH_SIZE = 24

def get_image_from_document(score_name):
    """ Loads PDF file as a cropped PNG and deletes the associated lilypond and PDF files afterwards
    """
    score_paths = {"pdf": Path(f"{score_name}.pdf")}
    for suffix in [".ly", ".log"]:
        score_paths[suffix] = Path(f"{score_name}{suffix}")

    document = fitz.open(score_paths["pdf"])
    pixel_map = document[0].get_pixmap(colorspace=fitz.csGRAY)
    image = Image.frombytes("L", [pixel_map.width, pixel_map.height], pixel_map.samples)
    crop_selector = (0, 0, image.size[0], image.size[0])
    cropped_image = image.crop(crop_selector)

    # Delete temp files
    document.close()
    for path in score_paths.values():
        if path.exists():
            path.unlink()
    
    return image


IMAGE_SHAPE = (512, 512)
IMAGE_MEAN = 0.5
IMAGE_STANDARD_DEVIATION = 0.5
COMPOSE_TRANSFORMS = [
    transforms.ToTensor(), 
    transforms.Resize(IMAGE_SHAPE), 
    transforms.Normalize(
        (IMAGE_MEAN), 
        (IMAGE_STANDARD_DEVIATION)
    )
]

class BrazenAbcHandler(base_handler.BaseHandler):
    """ Handle text in ABC format - convert to lilypond, render and evaluate
    """

    def initialize(self, context):
        super().initialize(context)
        self.transforms = transforms.Compose(COMPOSE_TRANSFORMS)

        #symposium = symposium.Symposium(self.model.config)
        #self.token_map = symposium.token_map

    def preprocess(self, data):
        line = data[0]
        text = line.get("data")

        if not text:
            raise ValueError("No text in request. Pass in a POST request with a 'data' field.")

        if isinstance(text, (bytes, bytearray)):
            text = text.decode('utf-8')

        filename = "output"
        with open(f"{filename}.abc", "w") as f:
            f.write(text)

        command = ["abc2ly", f"{filename}.abc"]
        subprocess.run(command, capture_output=True)
        next_command = ["lilypond", f"{filename}.ly"]
        subprocess.run(next_command, capture_output=True)

        image = get_image_from_document(filename)
        transformed_image = self.transforms(image)
        final = einops.repeat(torch.squeeze(transformed_image, 0), "height width -> batch height width", batch=BATCH_SIZE)

        return final
    
    def inference(self, data, *args, **kwargs):
        return super().inference(data, *args, **kwargs)

    def postprocess(self, data):
        print(data)
        _, label_indices = torch.max(data, dim=-1)
        return label_indices
        #output_labels, _ = train.generate_label(label_indices)
        #return output_labels