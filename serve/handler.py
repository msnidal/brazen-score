# https://github.com/pytorch/serve/tree/master/model-archiver
from cProfile import label
from xml.sax.handler import property_interning_dict
import torch
import torch.nn.functional as F
from torchvision import transforms
import subprocess
from pathlib import Path

import fitz
import pickle
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

def generate_label(label_indices, token_map):
    output_labels = []
    num_symbols = len(token_map)
    for batch in label_indices:
        batch_labels = []
        for index in batch:
            if index == num_symbols + 1:
                batch_labels.append("EOS")
            elif index == num_symbols:
                batch_labels.append("BOS")
            elif index == num_symbols + 2:
                batch_labels.append("")
            else:
                batch_labels.append(token_map[index])
        output_labels.append(batch_labels)

    return output_labels

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
DATASET_PROPERTIES_PATH = Path("/home/model-server/symposium_properties.pickle")

class BrazenAbcHandler(base_handler.BaseHandler):
    """ Handle text in ABC format - convert to lilypond, render and evaluate
    """

    def initialize(self, context):
        super().initialize(context)
        self.transforms = transforms.Compose(COMPOSE_TRANSFORMS)

        with open(str(DATASET_PROPERTIES_PATH), "rb") as handle:
            properties = pickle.load(handle)

        self.token_map = properties["tokens"]
        self.max_label_length = properties["max_label_length"] + 10

    def preprocess(self, data):
        print(data)
        line = data[0]
        body = line.get("body")
        if not body:
            raise ValueError("No body in request. Pass in a POST request with JSON.")
        
        score = body.get("score")
        if not score:
            raise ValueError("No score in JSON body. Pass in the ABC score with the score key in the body.")

        if isinstance(score, (bytes, bytearray)):
            score = score.decode('utf-8')

        filename = "output"
        with open(f"{filename}.abc", "w") as f:
            f.write(score)

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
        _, label_indices = torch.max(data[0], dim=-1)
        output_labels = generate_label(label_indices, self.token_map)
        return [" ".join(output_labels[0])]