from typing import Iterable
from grpc.aio import ServicerContext
from google.protobuf.json_format import MessageToDict

from pbs.speech2text_pb2_grpc import GowajeeSpeechToTextServicer
from pbs.speech2text_pb2 import TranscriptionResult, TranscribeConfig, \
                                TranscribeRequest, TranscribeResponse, \
                                StreamingTranscribeConfig, StreamingTranscribeRequest, \
                                StreamingTranscribeResponse, WordInfo

import grpc
import numpy as np
from pydub import AudioSegment
from omegaconf import OmegaConf
from recognizer import SpeechRecognizer
from utils import audiosegment_to_librosawav, validate_input, DecodeType

import soxr

MAX_BUFFER = 81920 # 8192*2*5
# MAX_BUFFER = 49152 # 8192*2*3
class GowajeeSpeechRecognizerService(GowajeeSpeechToTextServicer):
    def __init__(self, config_path: str = "local_configs.yaml"):
        configs = OmegaConf.load(config_path)

        self.recognizer = SpeechRecognizer(configs.recognizer)

    def Transcribe(self, 
                   request: TranscribeRequest,
                   context: ServicerContext
                   ) -> TranscribeResponse:
        config = MessageToDict(request.config)
        decoder_type =  config.get("decoderType", "LMBeamSearch")
        get_timestamps =  config.get("getWordTimestamps", False)
        get_speaking_rate = config.get("getSpeakingRate", False)
        word_list = config.get("wordList", None)

        audio_data = request.audio_data
        temp_file, error_message = validate_input(audio_data, decoder_type)
        if error_message!='':
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(error_message)
            return TranscribeResponse()

        sound = AudioSegment.from_file(temp_file).set_channels(1).set_frame_rate(8000)
        signal_array = audiosegment_to_librosawav(sound)

        results = self.recognizer.infer(signal_array, 
                                        8000,
                                        DecodeType[decoder_type], 
                                        get_timestamps=get_timestamps, 
                                        get_speak_rate=get_speaking_rate,
                                        hotwords=word_list)


        return TranscribeResponse(
            results=[TranscriptionResult(
                transcript=result.get("transcript", None),
                start_time=result.get("start_time", None),
                end_time=result.get("end_time", None),
                speaking_rate=result.get("speaking_rate", None),
                word_timestamps=[WordInfo(
                    word=item.get("word", None),
                    start_time=item.get("start_time", None),
                    end_time=item.get("end_time", None),
                    confidence=item.get("confidence", None),
                    ) 
                    for item in result.get("word_timestamps", [])]
            ) for result in results]
        )

    def StreamingTranscribe(self, 
                                 request_iterator: Iterable[StreamingTranscribeRequest], 
                                 context: ServicerContext
                                 ) -> Iterable[StreamingTranscribeResponse]:

        audio_buffer = b''
        prev_results = list()
        offset_time = 0
        for request in request_iterator:
            config = MessageToDict(request.streaming_config)
            transcribe_config = config.get("transcribeConfig", {})
            get_timestamps =  transcribe_config.get("getWordTimestamps", False)
            get_speaking_rate = transcribe_config.get("getSpeakingRate", False)
            word_list = transcribe_config.get("wordList", None)

            if len(audio_buffer) > MAX_BUFFER:
                if len(results) > 1:
                    last_segment = results[-1]
                    idx = int((last_segment["start_time"] - offset_time ) * config.get("sampleRate") * 2) 
                    audio_buffer = audio_buffer[idx:]
                    prev_results += results[:-1]
                    offset_time = prev_results[-1]["end_time"]
                else:
                    audio_buffer = b''
                    prev_results += results
                    offset_time = prev_results[-1]["end_time"]

            if len(request.audio_data) == 0:
                continue
 
            audio_buffer += request.audio_data
            try:
                print("audio_buffer", len(audio_buffer))
                signal_array = np.frombuffer(
                    audio_buffer, 
                    dtype=np.int16).reshape(-1)
                ##################################

                # print(signal_array)
                results = self.recognizer.infer(
                    signal_array, 
                    sample_rate=config.get("sampleRate"),
                    decoder_type=transcribe_config.get("decoderType", "LMBeamSearch"),
                    get_timestamps=True,
                    get_speak_rate=get_speaking_rate,
                    hotwords=word_list)
                print(results)
            except:
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
            
            yield StreamingTranscribeResponse(
                results=[TranscriptionResult(
                    transcript=result.get("transcript", None),
                    start_time=result.get("start_time", None),
                    end_time=result.get("end_time", None),
                    speaking_rate=result.get("speaking_rate", None),
                    word_timestamps=[WordInfo(
                        word=item.get("word", None),
                        start_time=item.get("start_time", None),
                        end_time=item.get("end_time", None),
                        confidence=item.get("confidence", None),
                        ) 
                        for item in result.get("word_timestamps", []) ] if get_timestamps else None
                ) for result in prev_results + results]
            )
            # config: TranscribeConfig = request_iterator.config

        # pass