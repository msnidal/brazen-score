import random
import string
from tokenize import String 
from pathlib import Path
import hashlib
import functools
import pickle5 as pickle

from PIL import Image
import fitz

import torch
from torchvision import transforms as tvtransforms
import abjad
from abjad.io import Illustrator
import dataset
import train

from dataset import PadToLargest
import parameters

# Measure choices from https://abjad.github.io/examples/corpus-selection.html
MEASURE_FRAGMENTS_FILENAME = "symposium_fragments.pickle"

PITCHES = string.ascii_uppercase[:7]
ACCIDENTALS = ["", "b", "#"]
CLEFS = ["treble", "bass"]
TIME_SIGNATURES = {"numerator": (2, 3, 4, 8), "denominator": (2, 4, 8)}
TRANSPOSE_RANGE = [-5, 5]
MEASURE_DISTRIBUTION = {"mean": 6, "sigma": 4, "min": 1, "max": 8}
DURATIONS = {1: "whole", 2: "half", 4: "quarter", 8: "eighth", 16: "sixteenth", 32: "thirty_second"}
OUTPUT_DIRECTORY = "scores"
DATASET_PROPERTIES_PATH = Path(f"symposium_properties.pickle")
STAFF_SIZES = [x for x in range(23, 27)]

FRAGMENT_RATIO_RANGE = (0.4, 0.7)
REST_RATIO_RANGE = (0.2, 0.4)


class Symposium(torch.utils.data.IterableDataset):
    def __init__(self, config:parameters.BrazenParameters, output_directory:Path=Path(OUTPUT_DIRECTORY), seed:int=None, transforms=None):
        """ Note: mutates config
        """
        self.output_directory = output_directory
        if transforms is None:
            transforms = []

        compose_transforms = [
            tvtransforms.ToTensor(), 
            tvtransforms.Resize(config.image_shape), 
            tvtransforms.Normalize(
                (config.image_mean), 
                (config.image_standard_deviation)
            )
        ] + transforms
        self.transforms = tvtransforms.Compose(compose_transforms)

        self.random = random.Random(seed)

        with open(MEASURE_FRAGMENTS_FILENAME, "rb") as file:
            self.measure_fragments = pickle.load(file)

        self.token_map, self.max_label_length = self.get_dataset_properties()
        config.set_dataset_properties(len(self.token_map), self.max_label_length)
        self.config = config


    def __iter__(self):
        """ Implement the dataset as an iterator - see __next__
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: # single process
            print("Single worker mode, using manually specified seed")
            return self
        else:
            print(f"Setting worker {worker_info.id} to seed {worker_info.seed}")
            self.random = random.Random(worker_info.seed)
            return self
    
    def set_seed(self, seed:int = 0):
        self.random = random.Random(seed)

    def generate_fragment(self, clef=CLEFS[0], name="measure", rest_threshold=0.2):
        """ Generate a fragment of a score, consisting of 1-8 notes in a jaunty tune
        """
        count_notes = self.random.randint(1, 4)
        notes = []
        for _ in range(count_notes):
            pitch = self.random.choice(PITCHES)
            accidental = self.random.choice(ACCIDENTALS)
            duration = self.random.choice(list(DURATIONS.keys()))

            if self.random.uniform(0, 1) < rest_threshold:
                note = abjad.Rest((1, duration))
            else:
                note = abjad.Note.from_pitch_and_duration(f"{pitch}{accidental}", (1, duration))
            notes.append(note)
        
        return abjad.Container(notes, name=name)
            
    def extract_timestamp(self, file_name, identifier_token):
        """ Extract a timestamp from an abjad-generated score filename """
        timestamp = file_name.split(f".{identifier_token}.pdf")[0]
        return timestamp

    def get_transpose_sequence(self, num_measures:int):
        f""" Get a transpose sequence of length in the range {TRANSPOSE_RANGE} of the specified length
        """
        transpose_sequence = []
        current_transpose = 0
        # Should be a relatively smooth random sequence but bounded
        for i in range(num_measures):
            current_transpose += self.random.randint(-2, 2)
            current_transpose = min(TRANSPOSE_RANGE[1], current_transpose)
            current_transpose = max(TRANSPOSE_RANGE[0], current_transpose)

            transpose_sequence.append(current_transpose)

        return transpose_sequence

    def get_label_token(self, leaf):
        """ Process an abjad leaf (note or rest) to get the label token
        """
        if type(leaf) == abjad.Note:
            label = leaf.written_pitch.pitch_class.pitch_class_label
            octave = leaf.written_pitch.octave.number
            duration = DURATIONS[leaf.written_duration.denominator]

            token = f"note-{label}{octave}_{duration}"
        elif type(leaf) == abjad.Chord:
            token_segments = []
            for pitch in leaf.written_pitches:
                label = pitch.pitch_class.pitch_class_label
                octave = pitch.octave.number

                token_segments.append(f"{label}{octave}")

            chord_token = "_".join(token_segments)
            duration = DURATIONS[leaf.written_duration.denominator]
            token = f"chord-{chord_token}_{duration}"
        elif type(leaf) == abjad.Rest:
            duration = DURATIONS[leaf.written_duration.denominator]
            assert leaf.written_duration.numerator == 1, "Not sure how to handle anything else, but shouldn't happen"
            token = f"rest-{duration}"
        else:
            raise Exception("Unknown abjad symbol type called")
        
        return token
    
    def get_score_image(self, score, config):
        """ Convert the score to a torch readable grayscale tensor format
        """
        illustrator = Illustrator(score, output_directory=self.output_directory, should_open=False)

        # Inject staff size manually
        score_string = illustrator.get_string()
        injected_string = "#(set-global-staff-size " + str(config["staff_size"]) + ")\n    " 

        insert_before = "\\context Score"
        insertion_index = score_string.find(insert_before)

        modified_string = score_string[:insertion_index] + injected_string + score_string[insertion_index:]
        illustrator.string = modified_string

        paths, format_time, render_time, success, log = illustrator()

        score_paths = {"pdf": paths[0]}
        for suffix in [".ly", ".log"]:
            score_paths[suffix] = score_paths["pdf"].with_suffix(suffix)

        if success is False:
            raise Exception("Could not render score")

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

        transformed_image = self.transforms(cropped_image)
        transformed_image = torch.squeeze(transformed_image, 0)

        return transformed_image
    
    def get_label_indices(self, label):
        """ Map the label to the indices in the token map
        """
        label_indices = [self.token_map.index(token) for token in label]
        length_pad = [self.config.end_of_sequence] + [self.config.padding_symbol for _ in range(self.max_label_length - len(label))]
        label_indices += length_pad

        label = torch.tensor(label_indices, dtype=torch.long)
        return label
    
    def generate_config(self):
        """ Pick all of the random parameters that define a unique score
        """
        config = {
            "key_mode": "major", # self.random.choice(["major", "minor"]),
            "key_pitch": self.random.choice(PITCHES),
            "key_accidental": self.random.choice(ACCIDENTALS),
            "time_signature": (self.random.choice(TIME_SIGNATURES["numerator"]), self.random.choice(TIME_SIGNATURES["denominator"])),
            "transpose_sequence": self.random.choice(TRANSPOSE_RANGE),
            "num_measures": min(MEASURE_DISTRIBUTION["max"], max(MEASURE_DISTRIBUTION["min"], round(self.random.normalvariate(MEASURE_DISTRIBUTION["mean"], MEASURE_DISTRIBUTION["sigma"])))),
            "clef": self.random.choice(CLEFS),
            "fragment_ratio": self.random.uniform(FRAGMENT_RATIO_RANGE[0], FRAGMENT_RATIO_RANGE[1]),
            "rest_ratio": self.random.uniform(REST_RATIO_RANGE[0], REST_RATIO_RANGE[1]),
            "staff_size": self.random.choice(STAFF_SIZES)
        }

        return config
    

    def get_score(self, config:dict):
        """ Get an abjad score with a label in string format based on the parameters from the passed configuration
        """

        # Pick some random fragments
        measures = []

        for index in range(config["num_measures"]):
            if random.uniform(0, 1) < config["fragment_ratio"]:
                measures.append(self.generate_fragment(clef=config["clef"], name=f"measure_{index}", rest_threshold=config["rest_ratio"]))
            else:
                measures.append(abjad.Container(self.random.choice(self.measure_fragments[config["clef"]]), name=f"measure_{index}"))

        # Transpose measures in-place
        transpose_sequence = self.get_transpose_sequence(config["num_measures"])
        for index, measure in enumerate(measures):
            abjad.mutate.transpose(measure, transpose_sequence[index])

        # transpose each measure differently to increase domain
        key_signature = abjad.KeySignature(abjad.NamedPitchClass(config["key_pitch"] + config["key_accidental"]), abjad.Mode(config["key_mode"]))
        time_signature = abjad.TimeSignature(config["time_signature"])

        voice = abjad.Voice(name="voice")
        staff = abjad.Staff([voice], name="staff")
        score = abjad.Score([staff], name="score")

        for measure in measures:
            voice.extend(measure)
        
        voice_start = abjad.get.leaf(voice, 0)
        clef = abjad.Clef(config["clef"])
        abjad.attach(clef, voice_start)
        abjad.attach(time_signature, voice_start)
        abjad.attach(key_signature, voice_start)

        label = []
        label.append("clef-G1" if config["clef"] == "treble" else "clef-F4")
        label.append(f"keySignature-{config['key_pitch']}{config['key_accidental']}M")
        label.append(f"timeSignature-{config['time_signature'][0]}/{config['time_signature'][1]}")

        time_accumulator = abjad.Duration(0)
        signature_duration = abjad.Duration(config["time_signature"])

        for leaf in voice:
            token = self.get_label_token(leaf)
            label.append(token)

            time_accumulator += leaf.written_duration
            if time_accumulator % signature_duration == abjad.Duration(0):
                label.append("barline")
        
        return score, label

    def __next__(self):
        """ Generate a random score using Mozart's dice game 
        For more details see https://abjad.github.io/examples/corpus-selection.html
        """

        config = self.generate_config()
        score, label = self.get_score(config)
        image = self.get_score_image(score, config)
        label_indices = self.get_label_indices(label)

        return image, label_indices
    

    def get_dataset_properties(self):
        """Load or create the token mapping if it does not already exist.
        Needs to be re-run if the symposium dataset properties are changed.
        """

        if DATASET_PROPERTIES_PATH.exists():
            with open(str(DATASET_PROPERTIES_PATH), "rb") as handle:
                properties = pickle.load(handle)
        else:
            properties = {"tokens": [], "max_label_length": 0}
            for _ in range(20000):
                _, label = self.get_score()
                properties["max_label_length"] = max(len(label), properties["max_label_length"])
                for token in label:
                    if token not in properties["tokens"]:
                        properties["tokens"].append(token)

            with open(str(DATASET_PROPERTIES_PATH), "wb") as handle:
                pickle.dump(properties, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return properties["tokens"], properties["max_label_length"] + 10


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    config = parameters.BrazenParameters()
    symposium = Symposium(config)
    score, label = next(symposium)
    plt.imshow(score)

    printed_label = [] 
    for token in label:
        if token < len(symposium.token_map):
            printed_label.append(symposium.token_map[token])

    print(" ".join(printed_label))