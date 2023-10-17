from typing import Any, Dict, List, Tuple, Union
import onnxruntime as ort

import numpy as np
import re

import librosa
from segmenter import n_segmenter
from decoder import Decoder

from cem import WordConfidenceEstimator
from g2p import ThaiG2P, EngG2P

import time
import logging

def normalize_signal(x):
    if max(x) > 1 or min(x) < -1:
        return x/32768.0
    return x
    
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1).reshape([x.shape[0], 1])

class SpeechRecognizer:
    def __init__(self, configs):
        self.configs = configs

        self.sess = ort.InferenceSession(configs.model_path)
        self.decoder = Decoder(configs.vocab, 
                               configs.decoder.beam_size, 
                               configs.decoder.char_lm_path, 
                               configs.decoder.word_lm_path, 
                               cutoff_prob=configs.decoder.cutoff_prob,
                               alpha=configs.decoder.alpha, 
                               beta=configs.decoder.beta, 
                               scale_unk=configs.decoder.scale_unk, 
                               word_boundary_token=configs.decoder.word_boundary_token)
        
        self.sg = n_segmenter.Segmenter(
            energy_threshold=configs.segmenter.energy_threshold,
            max_gap=configs.segmenter.max_gap, 
            max_duration=configs.segmenter.max_duration)

        self.thai_g2p = ThaiG2P()
        self.eng_g2p = EngG2P()

        self.cem = WordConfidenceEstimator(configs.cem)

    def infer(self, audio: np.array, sample_rate: int, 
                    decoder_type: int, get_timestamps: bool=False, 
                    get_speak_rate: bool=False, 
                    hotwords: Union[None, List[str]]=None) -> Dict:
        """
        Arguments:
            audio: array of audio signal.
            sample_rate: audio's sample rate should be greater or equal to 8 kHz.
            decoder_type: 0 for Greedy, 1 for BeamSearch, 2 for LMBeamSearch.
            get_timestamps: if 'True', the results will contains word-level timestamps. Default: False.
            get_speaker_rate: if 'True', the results will contains speaking rate (syllable/sec). Default: False.
            hotwords: List of words that gain apprerance
        """

        # set hotwords
        if hotwords is not None:
            self.decoder.scorer.set_hotwords(set(hotwords))
        else:
            self.decoder.scorer.set_hotwords(None)

        feats, time_slices = self.preprocessing(audio, sample_rate)
        if feats is None and time_slices is None:
            return None
        
        # LMBeamSearch is default
        beam_search = True
        use_lm = True
        if decoder_type == 0:
            beam_search = False
            use_lm = False
        elif decoder_type == 1:
            beam_search = True
            use_lm = False


        logits = [self.sess.run(None, {'audio_signal': 
                                    np.expand_dims(feat, 0)})[0][0] for feat in feats]
        all_probs = [softmax(output) for output in logits]

        outputs = [self.decoder.decode(probs, 
                                    beam_search=beam_search, 
                                    use_lm=use_lm) 
                for probs in all_probs
                ]

        # make dict results
        results = list()
        for i, ((text, frames), time_slice) in enumerate(zip(outputs, time_slices)):
            
            # remove duplicates word boundary token
            text = re.sub(' {2,}',' ', text.replace(
                self.configs.decoder.word_boundary_token, 
                ' ').strip())

            if text == '':
                continue
            
            # # remove spaces between thai words
            out_text = re.sub('(?<=[^a-z])\s(?=[^a-z])', '', text)
            # out_text = text

            result = {
                'transcript': out_text,
                'start_time':time_slice[0],
                'end_time':time_slice[1],
            }
            
            # add speaking rate
            if get_speak_rate:
                words = text.split()
                phn = []
                for word in words:
                    if re.search('[a-zA-Z]+',word) is not None:
                        phn += list(self.eng_g2p(word))
                    else:
                        phn += list(self.thai_g2p(word))
                phn = list(filter(lambda x: re.search('\d+', x) is not None, phn))
                result['speaking_rate'] = len(phn)/(time_slice[1]-time_slice[0])

            # add word timestamp
            if get_timestamps:
                _time_stamps=self.covert_frame_to_time(text.split(), frames, time_slice)

                pred, pred_prob = self.extract_logits(logits[i], text, frames)
                confidence = self.cem.infer(pred, np.expand_dims(pred_prob, 0))

                for i in range(len(_time_stamps)):
                    _time_stamps[i]['confidence'] = float(confidence[0][i][0])
                    
                result['word_timestamps'] = _time_stamps

            results.append(result)
            
        return results

    def extract_logits(self, logits, text, frame_info):
        T, C =  logits.shape

        space_index = C - 2
        words = text.strip().split()
        assert len(words) == len(frame_info)
        preds = []
        space_logits = np.zeros((1, C), dtype=np.float32)
        space_logits[:,space_index] = 1

        select_logits = []
        for word, frame in zip(words, frame_info):
            preds += [self.configs.vocab.index(c) for c in word] + [space_index]
            select_logits.append(logits[frame])
            select_logits.append(space_logits)
        return np.array([preds]), np.concatenate(select_logits, axis=0)

    def preprocessing(self, audio_signal, 
                            sample_rate: int
                            ) -> Union[Tuple[None, None], List[Tuple[float, float]]]:

        audio_signal = normalize_signal(audio_signal).astype(np.float32)

        # check resample
        if sample_rate != self.configs.preprocessing.sample_rate:
            audio_signal = librosa.resample(
                audio_signal, orig_sr=sample_rate, res_type="fft",
                target_sr=self.configs.preprocessing.sample_rate)
        
        # VAD
        try:
            results = self.sg(audio_signal, self.configs.preprocessing.sample_rate)
        except:
            results = None

        # error
        if results is None:
            return [self.get_melspectrogram(audio_signal)], [(0, len(audio_signal)/self.configs.preprocessing.sample_rate)]
        
        audio_list = [
            audio_signal[int(self.configs.preprocessing.sample_rate*st):int(self.configs.preprocessing.sample_rate*ed)] 
            for (st, ed) in results.signal_slices]

        mel_list = [self.get_melspectrogram(audio) for audio in audio_list]

        return mel_list, results.signal_slices

    def get_melspectrogram(self, audio):
        """
        melspectrogram extraction from Nvidia Nemo in numpy version
        """
        pre_audio = librosa.effects.preemphasis(audio, zi=0)
        mel = librosa.feature.melspectrogram(y=pre_audio, 
                                             sr=self.configs.preprocessing.sample_rate, 
                                             n_fft=self.configs.preprocessing.n_fft, 
                                             win_length=self.configs.preprocessing.win_length, 
                                             hop_length=self.configs.preprocessing.hop_length, 
                                             n_mels=self.configs.preprocessing.num_mels,)
        mel = np.log(mel+ (2 ** -24))
        mean, std = np.mean(mel, axis=1), np.std(mel, axis=1)
        pad_amt = mel.shape[-1] % 16
        mel = (mel - np.expand_dims(mean, 1))/np.expand_dims(std,1)
        mel = np.pad(mel, ((0, 0), (0, 16-pad_amt)), 'constant', constant_values=(0))
        return mel

    def covert_frame_to_time(self, words: List[str], 
                                   frames: List[List[int]], 
                                   time_slice: List[Tuple[float, float]]) -> Dict:
        """
        Convert frame number to time
        """
        st, ed = time_slice
        result = []
        for i, frame in enumerate(frames):
            w_st = frame[0]
            w_ed = frame[-1]
            real_w_st = ((w_st*self.configs.decoder.time_scale*self.configs.preprocessing.hop_length/self.configs.preprocessing.sample_rate)+st)
            real_w_ed = ((w_ed*self.configs.decoder.time_scale*self.configs.preprocessing.hop_length/self.configs.preprocessing.sample_rate)+st)
            
            if real_w_ed > ed:
                real_w_ed = ed
            result.append(
                {
                    'word': words[i],
                    'start_time': real_w_st,
                    'end_time': real_w_ed,
                }
            )
        return result
