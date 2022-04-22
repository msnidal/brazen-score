import random
import string 

import abjad

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

PITCHES = string.ascii_uppercase[:8]
ACCIDENTALS = ["", "b", "#"]
TIME_SIGNATURES = ((4, 4), (3, 8), (2, 8), (5, 8), (6, 8), (7, 8), (9, 8), (12, 8))
TRANSPOSE_RANGE = [-5, 5]
NUM_MEASURES = [4, 8]

def generate_fragment():
    """ Generate a fragment of a score, consisting of 1-8 notes in a jaunty tune
    """

    count_notes = random.randint(1, 8)
    notes = []
    for note_index in range(count_notes):
        pitch = random.choice(PITCHES)
        accidental = random.choice(ACCIDENTALS)
        note = abjad.Note(f"{pitch}{accidental}")
        notes.append(note)
    
    return notes
        

def get_transpose_sequence(num_measures:int=NUM_MEASURES[0]):
    f""" Get a transpose sequence of length in the range {TRANSPOSE_RANGE} of the specified length
    """

    transpose_sequence = []
    current_transpose = 0
    # Should be a relatively smooth random sequence but bounded
    for i in range(num_measures):
        current_transpose += random.randint(-2, 2)
        current_transpose = min(TRANSPOSE_RANGE[1], current_transpose)
        current_transpose = max(TRANSPOSE_RANGE[0], current_transpose)

        transpose_sequence.append(current_transpose)

    return transpose_sequence


class Symposium:
    def __init__(self):
        pass

    def generate_score(self, seed:int=None):
        """ Generate a random score using Mozart's dice game 
        For more details see https://abjad.github.io/examples/corpus-selection.html
        """
        random.seed(seed)

        config = {
            "key_mode": random.choice(["major", "minor"]),
            "key_pitch": random.choice(PITCHES),
            "key_accidental": random.choice(ACCIDENTALS),
            "time_signature": random.choice(TIME_SIGNATURES),
            "transpose_sequence": random.choice(TRANSPOSE_RANGE),
            "num_measures": random.randint(NUM_MEASURES[0], NUM_MEASURES[1])
        }
        
        measure_choices = {
            "bass": [abjad.Container(measure[0]) for measure_group in MEASURE_CHOICES for measure in measure_group],
            "treble": [abjad.Container(measure[1]) for measure_group in MEASURE_CHOICES for measure in measure_group]
        }
        measures = [random.choice(measure_choices["treble"]) for _ in range(config["num_measures"])]

        transpose_sequence = get_transpose_sequence(config["num_measures"])
        transposed_measures = [abjad.mutate.transpose(measure, transpose_sequence[index]) for index, measure in enumerate(measures)]

        # transpose each measure differently to increase domain
        key_signature = abjad.KeySignature(abjad.NamedPitchClass(config["key_pitch"] + config["key_accidental"]), abjad.Mode(config["key_mode"]))
        time_signature = abjad.TimeSignature(config["time_signature"])

        voice = abjad.Voice()
        staff = abjad.Staff([voice])
        score = abjad.Score([staff])

        for measure in measures:
            voice.extend(measure)
        
        treble_start = abjad.get.leaf(voice, 0)
        abjad.attach(time_signature, treble_start)
        label = "" # TODO: need ot fill this out

        return score, label


if __name__ == "__main__":
    symposium = Symposium()
    score, label = symposium.generate_score(seed=0)
    abjad.show(score)