import sys
import time
import pyaudio
from six.moves import queue

import grpc
from pbs.audio_pb2 import AudioEncoding
from pbs import speech2text_pb2_grpc, speech2text_pb2, decoder_type_pb2
from google.protobuf.json_format import MessageToDict


# Audio recording parameters
RATE = 48000
CHUNK = 1000

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            # print("chunk", chunk)
            
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


def listen_print_loop(responses):

    for response in responses:
        res = MessageToDict(response)
        print(res)


def main():

    channel = grpc.insecure_channel('localhost:50051')
    client = speech2text_pb2_grpc.GowajeeSpeechToTextStub(channel)

    transcribe_config = speech2text_pb2.TranscribeConfig(
        get_word_timestamps=False,
        get_speaking_rate=False,
        decoder_type = decoder_type_pb2.LMBeamSearch,
    )

    stream_config = speech2text_pb2.StreamingTranscribeConfig(
        transcribe_config = transcribe_config,
        sample_rate=RATE,
        encoding=AudioEncoding.LINEAR_PCM,
        num_channels=0
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()

        requests = (
            speech2text_pb2.StreamingTranscribeRequest(
                streaming_config=stream_config,
                audio_data=content
            )
            for content in audio_generator
        )

        responses = client.StreamingTranscribe(requests)

    #     # Now, put the transcription responses to use.
        listen_print_loop(responses)

if __name__ == "__main__":
    main()