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
MEASURE_CHOICES = [
    [
        ("c4 r8", "e''8 c''8 g'8"),
        ("<c e>4 r8", "g'8 c''8 e''8"),
        ("<c e>4 r8", "g''8 ( e''8 c''8 )"),
        ("<c e>4 r8", "c''16 b'16 c''16 e''16 g'16 c''16"),
        ("<c e>4 r8", "c'''16 b''16 c'''16 g''16 e''16 c''16"),
        ("c4 r8", "e''16 d''16 e''16 g''16 c'''16 g''16"),
        ("<c e>4 r8", "g''8 f''16 e''16 d''16 c''16"),
        ("<c e>4 r8", "e''16 c''16 g''16 e''16 c'''16 g''16"),
        ("<c e>16 g16 <c e>16 g16 <c e>16 g16", "c''8 g'8 e''8"),
        ("<c e>4 r8", "g''8 c''8 e''8"),
        ("c8 c8 c8", "<e' c''>8 <e' c''>8 <e' c''>8"),
    ],
    [
        ("c4 r8", "e''8 c''8 g'8"),
        ("<c e>4 r8", "g'8 c''8 e''8"),
        ("<c e>4 r8", "g''8 e''8 c''8"),
        ("<e g>4 r8", "c''16 g'16 c''16 e''16 g'16 c''16"),
        ("<c e>4 r8", "c'''16 b''16 c'''16 g''16 e''16 c''16"),
        ("c4 r8", "e''16 d''16 e''16 g''16 c'''16 g''16"),
        ("<c e>4 r8", "g''8 f''16 e''16 d''16 c''16"),
        ("<c e>4 r8", "c''16 g'16 e''16 c''16 g''16 e''16"),
        ("<c e>4 r8", "c''8 g'8 e''8"),
        ("<c e>4 <c g>8", "g''8 c''8 e''8"),
        ("c8 c8 c8", "<e' c''>8 <e' c''>8 <e' c''>8"),
    ],
    [
        ("<b, g>4 g,8", "d''16 e''16 f''16 d''16 c''16 b'16"),
        ("g,4 r8", "b'8 d''8 g''8"),
        ("g,4 r8", "b'8 d''16 b'16 a'16 g'16"),
        ("<g b>4 r8", "f''8 d''8 b'8"),
        ("<b, d>4 r8", "g''16 fs''16 g''16 d''16 b'16 g'16"),
        ("<g b>4 r8", "f''16 e''16 f''16 d''16 c''16 b'16"),
        ("<g, g>4 <b, g>8", "b'16 c''16 d''16 e''16 f''16 d''16"),
        ("g8 g8 g8", "<b' d''>8 <b' d''>8 <b' d''>8"),
        ("g,4 r8", "b'16 c''16 d''16 b'16 a'16 g'16"),
        ("b,4 r8", "d''8 ( b'8 g'8 )"),
        ("g4 r8", "b'16 a'16 b'16 c''16 d''16 b'16"),
    ],
    [
        ("<c e>4 r8", "c''16 b'16 c''16 e''16 g'8"),
        ("c4 r8", "e''16 c''16 b'16 c''16 g'8"),
        ("<e g>4 r8", "c''8 ( g'8 e'8 )"),
        ("<e g>4 r8", "c''8 e''8 g'8"),
        ("<e g>4 r8", "c''16 b'16 c''16 g'16 e'16 c'16"),
        ("<c e>4 r8", "c''8 c''16 d''16 e''8"),
        ("c4 r8", "<c'' e''>8 <c'' e''>16 <d'' f''>16 <e'' g''>8"),
        ("<e g>4 r8", "c''8 e''16 c''16 g'8"),
        ("<e g>4 r8", "c''16 g'16 e''16 c''16 g''8"),
        ("<e g>4 r8", "c''8 e''16 c''16 g''8"),
        ("<e g>4 r8", "c''16 e''16 c''16 g'16 e'8"),
    ],
    [
        ("c4 r8", "fs''8 a''16 fs''16 d''16 fs''16"),
        ("c8 c8 c8", "<fs' d''>8 <d'' fs''>8 <fs'' a''>8"),
        ("c4 r8", "d''16 a'16 fs''16 d''16 a''16 fs''16"),
        ("c8 c8 c8", "<fs' d''>8 <fs' d''>8 <fs' d''>8"),
        ("c4 r8", "d''8 a'8 ^\\turn fs''8"),
        ("c4 r8", "d''16 cs''16 d''16 fs''16 a''16 fs''16"),
        ("<c a>4 <c a>8", "fs''8 a''8 d''8"),
        ("<c fs>8 <c fs>8 <c a>8", "a'8 a'16 d''16 fs''8"),
        ("c8 c8 c8", "<d'' fs''>8 <d'' fs''>8 <d'' fs''>8"),
        ("<c d>8 <c d>8 <c d>8", "fs''8 fs''16 d''16 a''8"),
        ("<c a>4 r8", "fs''16 d''16 a'16 a''16 fs''16 d''16"),
    ],
    [
        ("<b, d>8 <b, d>8 <b, d>8", "g''16 fs''16 g''16 b''16 d''8"),
        ("<b, d>4 r8", "g''8 b''16 g''16 d''16 b'16"),
        ("<b, d>4 r8", "g''8 b''8 d''8"),
        ("<b, g>4 r8", "a'8 fs'16 g'16 b'16 g''16"),
        ("<b, d>4 <b, g>8", "g''16 fs''16 g''16 d''16 b'16 g'16"),
        ("b,4 r8", "g''8 b''16 g''16 d''16 g''16"),
        ("<b, g>4 r8", "d''8 g''16 d''16 b'16 d''16"),
        ("<b, g>4 r8", "d''8 d''16 g''16 b''8"),
        ("<b, d>8 <b, d>8 <b, g>8", "a''16 g''16 fs''16 g''16 d''8"),
        ("<b, d>4 r8", "g''8 g''16 d''16 b''8"),
        ("<b, d>4 r8", "g''16 b''16 g''16 d''16 b'8"),
    ],
    [
        ("c8 d8 d,8", "e''16 c''16 b'16 a'16 g'16 fs'16"),
        ("c8 d8 d,8", "a'16 e''16 <b' d''>16 <a' c''>16 <g' b'>16 <fs' a'>16"),
        (
            "c8 d8 d,8",
            "<b' d''>16 ( <a' c''>16 ) <a' c''>16 ( <g' b'>16 ) <g' b'>16 ( <fs' a'>16 )",
        ),
        ("c8 d8 d,8", "e''16 g''16 d''16 c''16 b'16 a'16"),
        ("c8 d8 d,8", "a'16 e''16 d''16 g''16 fs''16 a''16"),
        ("c8 d8 d,8", "e''16 a''16 g''16 b''16 fs''16 a''16"),
        ("c8 d8 d,8", "c''16 e''16 g''16 d''16 a'16 fs''16"),
        ("c8 d8 d,8", "e''16 g''16 d''16 g''16 a'16 fs''16"),
        ("c8 d8 d,8", "e''16 c''16 b'16 g'16 a'16 fs'16"),
        ("c8 d8 d,8", "e''16 c'''16 b''16 g''16 a''16 fs''16"),
        ("c8 d8 d,8", "a'8 d''16 c''16 b'16 a'16"),
    ],
    [
        ("g,8 g16 f16 e16 d16", "<g' b' d'' g''>4 r8"),
        ("g,8 b16 g16 fs16 e16", "<g' b' d'' g''>4 r8"),
    ],
    [
        ("d4 c8", "fs''8 a''16 fs''16 d''16 fs''16"),
        ("<d fs>4 r8", "d''16 a'16 d''16 fs''16 a''16 fs''16"),
        ("<d a>8 <d fs>8 <c d>8", "fs''8 a''8 fs''8"),
        ("<c a>4 <c a>8", "fs''16 a''16 d'''16 a''16 fs''16 a''16"),
        ("d4 c8", "d'16 fs'16 a'16 d''16 fs''16 a''16"),
        ("d,16 d16 cs16 d16 c16 d16", "<a' d'' fs''>8 fs''4 ^\\trill"),
        ("<d fs>4 <c fs>8", "a''8 ( fs''8 d''8 )"),
        ("<d fs>4 <c fs>8", "d'''8 a''16 fs''16 d''16 a'16"),
        ("<d fs>4 r8", "d''16 a'16 d''8 fs''8"),
        ("<c a>4 <c a>8", "fs''16 d''16 a'8 fs''8"),
        ("<d fs>4 <c a>8", "a'8 d''8 fs''8"),
    ],
    [
        ("<b, g>4 r8", "g''8 b''16 g''16 d''8"),
        ("b,16 d16 g16 d16 b,16 g,16", "g''8 g'8 g'8"),
        ("b,4 r8", "g''16 b''16 g''16 b''16 d''8"),
        ("<b, d>4 <b, d>8", "a''16 g''16 b''16 g''16 d''16 g''16"),
        ("<b, d>4 <b, d>8", "g''8 d''16 b'16 g'8"),
        ("<b, d>4 <b, d>8", "g''16 b''16 d'''16 b''16 g''8"),
        ("<b, d>4 r8", "g''16 b''16 g''16 d''16 b'16 g'16"),
        ("<b, d>4 <b, d>8", "g''16 d''16 g''16 b''16 g''16 d''16"),
        ("<b, d>4 <b, g>8", "g''16 b''16 g''8 d''8"),
        ("g,16 b,16 g8 b,8", "g''8 d''4 ^\\trill"),
        ("b,4 r8", "g''8 b''16 d'''16 d''8"),
    ],
    [
        ("c16 e16 g16 e16 c'16 c16", "<c'' e''>8 <c'' e''>8 <c'' e''>8"),
        ("e4 e16 c16", "c''16 g'16 c''16 e''16 g''16 <c'' e''>16"),
        ("<c g>4 <c e>8", "e''8 g''16 e''16 c''8"),
        ("<c g>4 r8", "e''16 c''16 e''16 g''16 c'''16 g''16"),
        ("<c g>4 <c g>8", "e''16 g''16 c'''16 g''16 e''16 c''16"),
        ("c16 b,16 c16 d16 e16 fs16", "<g' c'' e''>8 e''4 ^\\trill"),
        ("<c e>16 g16 <c e>16 g16 <c e>16 g16", "e''8 c''8 g'8"),
        ("<c g>4 <c e>8", "e''8 c''16 e''16 g''16 c'''16"),
        ("<c g>4 <c e>8", "e''16 c''16 e''8 g''8"),
        ("<c g>4 <c g>8", "e''16 c''16 g'8 e''8"),
        ("<c g>4 <c e>8", "e''8 ( g''8 c'''8 )"),
    ],
    [
        ("g4 g,8", "<c'' e''>8 <b' d''>8 r8"),
        ("<g, g>4 g8", "d''16 b'16 g'8 r8"),
        ("g8 g,8 r8", "<c'' e''>8 <b' d''>16 <g' b'>16 g'8"),
        ("g4 r8", "e''16 c''16 d''16 b'16 g'8"),
        ("g8 g,8 r8", "g''16 e''16 d''16 b'16 g'8"),
        ("g4 g,8", "b'16 d''16 g''16 d''16 b'8"),
        ("g8 g,8 r8", "e''16 c''16 b'16 d''16 g''8"),
        ("<g b>4 r8", "d''16 b''16 g''16 d''16 b'8"),
        ("<b, g>4 <b, d>8", "d''16 b'16 g'8 g''8"),
        ("g16 fs16 g16 d16 b,16 g,16", "d''8 g'4"),
        ("g16 fs16 g16 d16 b,16 g,16", "d''8 g'4"),
    ],
    [
        ("<c e>16 g16 <c e>16 g16 <c e>16 g16", "e''8 c''8 g'8"),
        ("<c e>16 g16 <c e>16 g16 <c e>16 g16", "g'8 c''8 e''8"),
        ("<c e>16 g16 <c e>16 g16 <c e>16 g16", "g''8 e''8 c''8"),
        ("<c e>4 <e g>8", "c''16 b'16 c''16 e''16 g'16 c''16"),
        ("<c e>4 <c g>8", "c'''16 b''16 c'''16 g''16 e''16 c''16"),
        ("<c g>4 <c e>8", "e''16 d''16 e''16 g''16 c'''16 g''16"),
        ("<c e>4 r8", "g''8 f''16 e''16 d''16 c''16"),
        ("<c e>4 r8", "c''16 g'16 e''16 c''16 g''16 e''16"),
        ("<c e>16 g16 <c e>16 g16 <c e>16 g16", "c''8 g'8 e''8"),
        ("<c e>16 g16 <c e>16 g16 <c e>16 g16", "g''8 c''8 e''8"),
        ("c8 c8 c8", "<e' c''>8 <e' c''>8 <e' c''>8"),
    ],
    [
        ("<c e>16 g16 <c e>16 g16 <c e>16 g16", "e''8 ( c''8 g'8 )"),
        ("<c e>4 <c g>8", "g'8 ( c''8 e''8 )"),
        ("<c e>16 g16 <c e>16 g16 <c e>16 g16", "g''8 e''8 c''8"),
        ("<c e>4 <c e>8", "c''16 b'16 c''16 e''16 g'16 c''16"),
        ("<c e>4 r8", "c'''16 b''16 c'''16 g''16 e''16 c''16"),
        ("<c g>4 <c e>8", "e''16 d''16 e''16 g''16 c'''16 g''16"),
        ("<c e>4 <e g>8", "g''8 f''16 e''16 d''16 c''16"),
        ("<c e>4 r8", "c''16 g'16 e''16 c''16 g''16 e''16"),
        ("<c e>16 g16 <c e>16 g16 <c e>16 g16", "c''8 g'8 e''8"),
        ("<c e>16 g16 <c e>16 g16 <c e>16 g16", "g''8 c''8 e''8"),
        ("c8 c8 c8", "<e' c''>8 <e' c''>8 <e' c''>8"),
    ],
    [
        ("<f a>4 <g d'>8", "d''16 f''16 d''16 f''16 b'16 d''16"),
        ("f4 g8", "d''16 f''16 a''16 f''16 d''16 b'16"),
        ("f4 g8", "d''16 f''16 a'16 d''16 b'16 d''16"),
        ("f4 g8", "d''16 ( cs''16 ) d''16 f''16 g'16 b'16"),
        ("f8 d8 g8", "f''8 d''8 g''8"),
        ("f16 e16 d16 e16 f16 g16", "f''16 e''16 d''16 e''16 f''16 g''16"),
        ("f16 e16 d8 g8", "f''16 e''16 d''8 g''8"),
        ("f4 g8", "f''16 e''16 d''16 c''16 b'16 d''16"),
        ("f4 g8", "f''16 d''16 a'8 b'8"),
        ("f4 g8", "f''16 a''16 a'8 b'16 d''16"),
        ("f4 g8", "a'8 f''16 d''16 a'16 b'16"),
    ],
    [
        ("c8 g,8 c,8", "c''4 r8"),
        ("c4 c,8", "c''8 c'8 r8"),
    ]
]

PITCHES = string.ascii_uppercase[:7]
ACCIDENTALS = ["", "b", "#"]
CLEFS = ["treble", "bass"]
TIME_SIGNATURES = {"numerator": (2, 3, 4, 8), "denominator": (2, 4, 8)}
TRANSPOSE_RANGE = [-5, 5]
NUM_MEASURES = [4, 8]
DURATIONS = {1: "whole", 2: "half", 4: "quarter", 8: "eighth", 16: "sixteenth", 32: "thirty_second"}
OUTPUT_DIRECTORY = "scores"
DATASET_PROPERTIES_PATH = Path(f"symposium_properties.pickle")


class Symposium(torch.utils.data.IterableDataset):
    def __init__(self, config:parameters.BrazenParameters, output_directory:Path=Path(OUTPUT_DIRECTORY), seed:int=None, transforms=None):
        """ Note: mutates config
        """
        self.output_directory = output_directory
        if transforms is None:
            transforms = []

        transforms = [tvtransforms.ToTensor(), tvtransforms.Resize(config.image_shape)] + transforms
        self.transforms = tvtransforms.Compose(transforms)
        self.token_map, self.max_label_length = self.get_dataset_properties()
        config.set_dataset_properties(len(self.token_map), self.max_label_length)

        self.random = random.Random(seed)

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

    def generate_fragment(self):
        """ Generate a fragment of a score, consisting of 1-8 notes in a jaunty tune
        """
        count_notes = self.random.randint(1, 8)
        notes = []
        for _ in range(count_notes):
            pitch = self.random.choice(PITCHES)
            accidental = self.random.choice(ACCIDENTALS)
            note = abjad.Note(f"{pitch}{accidental}")
            notes.append(note)
        
        return notes
            
    def extract_timestamp(self, file_name, identifier_token):
        """ Extract a timestamp from an abjad-generated score filename """
        timestamp = file_name.split(f".{identifier_token}.pdf")[0]
        return timestamp

    def get_transpose_sequence(self, num_measures:int=NUM_MEASURES[0]):
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

    def get_score_image(self, score):
        """ Convert the score to a torch readable grayscale tensor format
        """
        illustrator = Illustrator(score, output_directory=self.output_directory, should_open=False)
        paths, format_time, render_time, success, log = illustrator()

        score_paths = {"pdf": paths[0]}
        for suffix in [".ly", ".log"]:
            score_paths[suffix] = score_paths["pdf"].with_suffix(suffix)

        if success is False:
            raise Exception("Could not render score")

        document = fitz.open(score_paths["pdf"])
        pixel_map = document[0].get_pixmap(colorspace=fitz.csGRAY)
        image = Image.frombytes("L", [pixel_map.width, pixel_map.height], pixel_map.samples)

        # Delete temp files
        document.close()
        for path in score_paths.values():
            if path.exists():
                path.unlink()

        image = self.transforms(image)
        image = torch.squeeze(image, 0)

        return image
    
    def get_label_indices(self, label):
        """ Map the label to the indices in the token map
        """
        label_indices = [self.token_map.index(token) for token in label]
        length_pad = [len(self.token_map)] + [len(self.token_map) + 1 for _ in range(self.max_label_length - len(label))]
        label_indices += length_pad

        label = torch.tensor(label_indices, dtype=torch.long)
        return label
    
    def get_score(self):
        """ Get an abjad score with a label in string format
        """

        config = {
            "key_mode": "major", # self.random.choice(["major", "minor"]),
            "key_pitch": self.random.choice(PITCHES),
            "key_accidental": self.random.choice(ACCIDENTALS),
            "time_signature": (self.random.choice(TIME_SIGNATURES["numerator"]), self.random.choice(TIME_SIGNATURES["denominator"])),
            "transpose_sequence": self.random.choice(TRANSPOSE_RANGE),
            "num_measures": self.random.randint(NUM_MEASURES[0], NUM_MEASURES[1]),
            "clef": self.random.choice(CLEFS)
        }
        
        measure_choices = {
            "treble": [measure[1] for measure_group in MEASURE_CHOICES for measure in measure_group],
            "bass": [measure[0] for measure_group in MEASURE_CHOICES for measure in measure_group]
        }
        measures = [abjad.Container(self.random.choice(measure_choices[config["clef"]]), name=f"measure_{index}") for index in range(config["num_measures"])]

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

        score, label = self.get_score()
        image = self.get_score_image(score)
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
    config = parameters.BrazenParameters()
    symposium = Symposium(config)
    score, label = next(symposium)
    print(label)