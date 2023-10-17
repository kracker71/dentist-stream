from lm.ctc_beamsearch import ctc_beamsearch
from lm.lm import MultiLevelLM
import numpy as np

def ctc_greedy_decoder(probs, vocabulary):
    token_ids = np.argmax(probs, axis=1)
    blank_id = len(probs[0])-1
    word_boundary_id = blank_id-1
    prev = blank_id
    
    frames = list()
    preds = list()
    last_frame = -1
    for i, token_id in enumerate(token_ids):
        if token_id == prev or token_id == blank_id:
            continue
        
        if token_id == word_boundary_id:
            frames.append([last_frame,i])
            last_frame = -1
        else:
            if last_frame < 0:
                last_frame = i
                
        preds.append(vocabulary[token_id])
        prev = token_id
        
    # check for last frame
    if last_frame > 0:
        frames.append([last_frame,len(probs)-1])
        
    return ''.join(preds), frames
    

class Decoder():

    def __init__(
        self, vocab, beam_width, char_lm_path, word_lm_path, 
        alpha=0.7, beta=0.1, cutoff_prob=1.0, scale_unk=0.05,
        word_boundary_token=' ', cutoff_top_n=40, hotwords=None
    ):

        if char_lm_path is not None and word_lm_path is not None:
            print("Character language model:", char_lm_path)
            print("Word language model:", word_lm_path)
            self.scorer = MultiLevelLM(char_lm_path, word_lm_path, 
                                       is_scoring_boundary=True, word_boundary_token=word_boundary_token, 
                                       alpha=alpha, beta=beta, scale_unk=scale_unk, hotword_weight = 15,
                                       hotwords=hotwords
                                    #    hotwords=['manager', 'driver', 'team', 'brand', 'test', 'coding', 'performance',
                                    #             'contract', 'post', 'job', 'personality', 'candidate', 'lead',
                                    #             'process', 'action', 'position', 'task', 'assign', 'timeline',
                                    #             'management', 'recuiter', 'ambassador', 'reflect',
                                    #             'programming', 'technical', 'intelligence', 'search']
                                       )
        else:
            self.scorer = None

        self.vocab = vocab
        self.beam_width = beam_width
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n
        self.word_boundary_token = word_boundary_token

    def decode(self, probs, beam_search=False, use_lm=False):

        if beam_search:
            text, frame, _ = ctc_beamsearch(probs, self.vocab, 
                                 lm_scorer=self.scorer if use_lm else None, 
                                 word_boundary_token=self.word_boundary_token,
                                 beamwidth=self.beam_width, 
                                 cutoff_prob=self.cutoff_prob, 
                                 cutoff_top=self.cutoff_top_n )[0]
        else:
            text, frame = ctc_greedy_decoder(probs, self.vocab)
        return text, frame
