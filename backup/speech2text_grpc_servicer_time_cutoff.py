from typing import Iterable
from grpc.aio import ServicerContext
from google.protobuf.json_format import MessageToDict

from pbs.speech2text_pb2_grpc import GowajeeSpeechToTextServicer
from pbs.speech2text_pb2 import (
    TranscriptionResult,
    TranscribeConfig,
    TranscribeRequest,
    TranscribeResponse,
    StreamingTranscribeConfig,
    StreamingTranscribeRequest,
    StreamingTranscribeResponse,
    WordInfo,
)

import grpc
import numpy as np
from pydub import AudioSegment
from omegaconf import OmegaConf
from recognizer import SpeechRecognizer
from utils import audiosegment_to_librosawav, validate_input, DecodeType

import soxr
import scipy.io.wavfile as wav

MAX_BUFFER = 81920 * 3  # 8192*2*5
# MAX_BUFFER = 49152 # 8192*2*3
class GowajeeSpeechRecognizerService(GowajeeSpeechToTextServicer):
    def __init__(self, config_path: str = "local_configs.yaml"):
        configs = OmegaConf.load(config_path)

        self.recognizer = SpeechRecognizer(configs.recognizer)

    def Transcribe(
        self, request: TranscribeRequest, context: ServicerContext
    ) -> TranscribeResponse:
        config = MessageToDict(request.config)
        decoder_type = config.get("decoderType", "LMBeamSearch")
        get_timestamps = config.get("getWordTimestamps", False)
        get_speaking_rate = config.get("getSpeakingRate", False)
        word_list = config.get("wordList", None)

        audio_data = request.audio_data
        temp_file, error_message = validate_input(audio_data, decoder_type)
        if error_message != "":
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(error_message)
            return TranscribeResponse()

        sound = AudioSegment.from_file(temp_file).set_channels(1).set_frame_rate(8000)
        signal_array = audiosegment_to_librosawav(sound)

        results = self.recognizer.infer(
            signal_array,
            8000,
            DecodeType[decoder_type],
            get_timestamps=get_timestamps,
            get_speak_rate=get_speaking_rate,
            hotwords=word_list,
        )

        return TranscribeResponse(
            results=[
                TranscriptionResult(
                    transcript=result.get("transcript", None),
                    start_time=result.get("start_time", None),
                    end_time=result.get("end_time", None),
                    speaking_rate=result.get("speaking_rate", None),
                    word_timestamps=[
                        WordInfo(
                            word=item.get("word", None),
                            start_time=item.get("start_time", None),
                            end_time=item.get("end_time", None),
                            confidence=item.get("confidence", None),
                        )
                        for item in result.get("word_timestamps", [])
                    ],
                )
                for result in results
            ]
        )

    def StreamingTranscribe(
        self,
        request_iterator: Iterable[StreamingTranscribeRequest],
        context: ServicerContext,
    ) -> Iterable[StreamingTranscribeResponse]:

        # print("Start") ## Debugging, Delete This line on production pharse
        audio_buffer = b""
        #### Add-on
        long_audio_buffer = b""
        ####
        prev_results = list()
        offset_time = 0
        for request in request_iterator:
            # print(request) ## Debugging, Delete This line on production pharse
            is_final_option = request.is_final
            config = MessageToDict(request.streaming_config)
            transcribe_config = config.get("transcribeConfig", {})
            get_timestamps = transcribe_config.get("getWordTimestamps", False)
            get_speaking_rate = transcribe_config.get("getSpeakingRate", False)
            word_list = transcribe_config.get("wordList", None)

            ### Add-on is_final: is_final will be true if the audio_buffer is over the capability
            is_final = False
            if len(audio_buffer) > MAX_BUFFER:
                is_final = True

                if len(results) > 1:
                    last_segment = results[-1]
                    idx = int(
                        (last_segment["start_time"] - offset_time)
                        * config.get("sampleRate")
                        * 2
                    )
                    audio_buffer = audio_buffer[idx:]
                    prev_results += results[:-1]
                    offset_time = prev_results[-1]["end_time"]
                    
                    final_response = self.create_response(results[:-1], get_timestamps, True)
                else:
                    audio_buffer = b""
                    prev_results += results
                    if prev_results: # prevent list out of range in the early state
                        offset_time = prev_results[-1]["end_time"]
                    else:
                        offset_time = 0
                    
                    final_response = self.create_response(results, get_timestamps, True)
                
                yield final_response

            if len(request.audio_data) == 0:
                continue

            #### Add-on
            long_audio_buffer += request.audio_data
            ## The more value, the less time that it will predict in one second.
            if len(long_audio_buffer) <= 29000:  # 15000, 12000:
                continue
            ####

            # audio_buffer += request.audio_data
            audio_buffer += long_audio_buffer
            long_audio_buffer = b""

            try:
                signal_array = np.frombuffer(audio_buffer, dtype=np.int16).reshape(-1)
                print("Config", config.get("sampleRate"))

                results = self.recognizer.infer(
                    signal_array,
                    sample_rate=config.get("sampleRate"),
                    decoder_type=transcribe_config.get("decoderType", "LMBeamSearch"),
                    get_timestamps=True,
                    get_speak_rate=get_speaking_rate,
                    hotwords=word_list,
                )
            except:
                print("ERROR")
                pass

            # fix time
            if len(prev_results) > 0:
                for i, result in enumerate(results):
                    result["start_time"] = result["start_time"] + offset_time
                    result["end_time"] = result["end_time"] + offset_time

                    if "word_timestamps" in result.keys():
                        for i, item in enumerate(result["word_timestamps"]):
                            item["start_time"] = item["start_time"] + offset_time
                            item["end_time"] = item["end_time"] + offset_time
                            result["word_timestamps"][i] = item

            if not is_final_option:
                yield StreamingTranscribeResponse(
                    results=[
                        TranscriptionResult(
                            transcript=result.get("transcript", None),
                            start_time=result.get("start_time", None),
                            end_time=result.get("end_time", None),
                            speaking_rate=result.get("speaking_rate", None),
                            word_timestamps=[
                                WordInfo(
                                    word=item.get("word", None),
                                    start_time=item.get("start_time", None),
                                    end_time=item.get("end_time", None),
                                    confidence=item.get("confidence", None),
                                )
                                for item in result.get("word_timestamps", [])
                            ]
                            if get_timestamps
                            else None,
                        )
                        for result in prev_results + results
                    ]
                )
            else:
                
                yield StreamingTranscribeResponse(
                    results=[
                        TranscriptionResult(
                            transcript=result.get("transcript", None),
                            start_time=result.get("start_time", None),
                            end_time=result.get("end_time", None),
                            speaking_rate=result.get("speaking_rate", None),
                            word_timestamps=[
                                WordInfo(
                                    word=item.get("word", None),
                                    start_time=item.get("start_time", None),
                                    end_time=item.get("end_time", None),
                                    confidence=item.get("confidence", None),
                                )
                                for item in result.get("word_timestamps", [])
                            ]
                            if get_timestamps
                            else None,
                        )
                        for result in results
                    ],
                    is_final = is_final
                )
            # config: TranscribeConfig = request_iterator.config

        # pass
        
    def create_response(self, transcription, get_timestamps, is_final=False):
        return StreamingTranscribeResponse(
                    results=[
                        TranscriptionResult(
                            transcript=result.get("transcript", None),
                            start_time=result.get("start_time", None),
                            end_time=result.get("end_time", None),
                            speaking_rate=result.get("speaking_rate", None),
                            word_timestamps=[
                                WordInfo(
                                    word=item.get("word", None),
                                    start_time=item.get("start_time", None),
                                    end_time=item.get("end_time", None),
                                    confidence=item.get("confidence", None),
                                )
                                for item in result.get("word_timestamps", [])
                            ]
                            if get_timestamps
                            else None,
                        )
                        for result in transcription
                    ],
                    is_final = is_final
                )
