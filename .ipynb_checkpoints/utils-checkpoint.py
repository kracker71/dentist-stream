import base64
import binascii
from io import BytesIO

from enum import Enum
from typing import Tuple

import numpy as np
from pydub import AudioSegment

class DecodeType(Enum):
    Greedy = 0
    BeamSearch = 1
    LMBeamSearch= 2
    
def validate_input(str_byte: str, decoder_type: str) -> Tuple["BytesIO", str]:
    error_message = ''
    temp_file = None

    if str_byte == '':
        error_message = 'No input audio data.'
    elif decoder_type not in DecodeType._member_names_:
        error_message = 'Invalid decode type.'
    else:
        try:
            if isinstance(str_byte, bytes):
                temp_file = BytesIO(str_byte)
            else:
                temp_file = BytesIO(base64.b64decode(str_byte))
        except binascii.Error:
            error_message = 'Input is not base64 format.'

    return temp_file, error_message

def audiosegment_to_librosawav(audiosegment: AudioSegment) ->  np.ndarray:    
    channel_sounds = audiosegment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    fp_arr = fp_arr.reshape(-1)

    return fp_arr