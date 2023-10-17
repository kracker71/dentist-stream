from math import log10
import kenlm
from pygtrie import CharTrie
from typing import Tuple

class LanguageModel:
    def __init__(self, kenlm_path :str, alpha :float= 5.0, beta :float= 0.1, 
                 oov_offset :float = -1000, word_boundary_token :str= ' '):
        self.model = kenlm.Model(kenlm_path)
        self.oov_offset = oov_offset
        self.word_boundary_token = word_boundary_token
        self.alpha = alpha
        self.beta = beta
        
    def score_state(self, token, state=None, is_last=False):
        raise NotImplementedError()
    
    @property
    def order(self):
        return self.model.order

class WordBasedLM(LanguageModel):
    def __init__(self, kenlm_path :str, alpha :float= 5.0, beta :float= 0.1, 
                 oov_offset :float = -1000, word_boundary_token :str= ' ', 
                 hotwords=None, hotword_weight=10.0):
        super(WordBasedLM, self).__init__(kenlm_path, alpha, beta, 
                                          oov_offset, word_boundary_token)
        self.hotword_weight = hotword_weight
        self.hotwords_tries = CharTrie()
        self.hotwords = hotwords
        self.set_hotwords(hotwords)
        
    def score_state(self, token: Tuple[str, str], state=None, is_last=False) -> Tuple[float, "kenlm.State"]:
        """
        if state == NULL : First state 
        """

        if state is None and is_last == True:
            raise ValueError("state must be not None if is_last is True")

        char_token, word_token = token

        if char_token != self.word_boundary_token:
            return 0, state
        else:
            score = 0
            next_state = kenlm.State()

            # start token
            if state is None:
                start_state = kenlm.State()
                self.model.BeginSentenceWrite(start_state)
                _ = self.model.BaseScore(start_state, '<s>', next_state)

                state = next_state
                next_state = kenlm.State()

            # check for OOV
            if word_token in self.model:
                wlm_score = self.model.BaseScore(state, word_token, next_state)
            
            if self.hotwords is not None:
                if word_token in self.hotwords:
                    wlm_score =  wlm_score + self.hotword_weight
                 
            # end token
            if is_last:
                end_state = kenlm.State()
                score = self.model.BaseScore(next_state, '</s>', end_state)
                next_state = end_state

        score = self.alpha*score + self.beta
        return score, next_state
    
    def set_hotwords(self, hotwords: set):
        if hotwords is not None:
            self.hotwords = set(hotwords)
            for w in set(hotwords):
                self.hotwords_tries[w] = 1
        else:
            self.hotwords = hotwords
            
    def score_sentence(self, text):
        return self.model.score(text)
    
class CharBasedLM(LanguageModel):
    def __init__(self, kenlm_path :str, alpha :float= 5.0, beta :float= 0.1, 
                 oov_offset :float = -1000, word_boundary_token :str= ' ', is_scoring_boundary=False):
        super(CharBasedLM, self).__init__(kenlm_path, alpha, beta, 
                                          oov_offset, word_boundary_token)
        
        self.is_scoring_boundary = is_scoring_boundary
        
    def score_state(self, token: Tuple[str, str], state=None, is_last=False) -> Tuple[float, "kenlm.State"]:
        """
        if state == NULL : First state 
        """
        if state is None and is_last == True:
            raise ValueError("state must be not None if is_last is True")

        char_token, _ = token
        if char_token == self.word_boundary_token and not self.is_scoring_boundary:
            return 0, state
        else:
            next_state = kenlm.State()
            
            # start token
            if state is None:
                state = kenlm.State()
                self.model.BeginSentenceWrite(state)
                _ = self.model.BaseScore(state, '<s>', next_state)
                state = next_state
                next_state = kenlm.State()
                
            # check for OOV
            if char_token not in self.model:
                _ = self.model.BaseScore(state, '<unk>', next_state)
                score = self.oov_offset
            else:
                score = self.model.BaseScore(state, char_token, next_state)
             
            # end
            if is_last:
                end_state = kenlm.State()
                score = self.model.BaseScore(next_state, '</s>', end_state)
                next_state = end_state

        score = self.alpha*score + self.beta
        return score, next_state
    
    def score_sentence(self, text):
        score = 0.0
        if self.is_scoring_boundary:
            score =  self.model.score(' '.join(list(text)))
        else:
            score = self.model.score(' '.join(list(text.replace(self.word_boundary_token, ''))))
        return self.alpha*score + self.beta
    
class MultiLevelLM(LanguageModel):
    def __init__(self, char_kenlm_path :str, word_kenlm_path :str, alpha :float= 5.0, beta :float= 0.1, 
                 oov_offset :float = -1000, word_boundary_token :str= ' ', scale_unk=0.01,
                 is_scoring_boundary=False, hotwords=None, hotword_weight=10.0,):

        super(MultiLevelLM, self).__init__(word_kenlm_path, alpha, beta, 
                                          oov_offset, word_boundary_token)
        
        self.char_model = kenlm.Model(char_kenlm_path)
        self.is_scoring_boundary = is_scoring_boundary
        self.scale_unk = scale_unk
        self.hotword_weight = hotword_weight
        self.hotwords_tries = CharTrie()
        self.hotwords = hotwords
        self.set_hotwords(hotwords)

    def score_state(self, token: Tuple[str, str], state=None, is_last=False) -> Tuple[float, "kenlm.State"]:
        """
        if state == NULL : First state 
        """

        if state is [None, None] and is_last == True:
            raise ValueError("state must be not None if is_last is True")

        char_token, word_token = token
        c_state, w_state = state
        clm_score = 0.0
        wlm_score = 0.0
        c_next_state = c_state
        w_next_state = w_state
        
        if char_token == self.word_boundary_token:
            
            if self.is_scoring_boundary:
                c_next_state = kenlm.State()
                clm_score = self.char_model.BaseScore(c_state, char_token, c_next_state)
            
            # scored_hw = False
            if word_token != '':
                w_next_state = kenlm.State()
                
                # start state
                if w_state is None:
                    w_state = kenlm.State()

                    self.model.BeginSentenceWrite(w_state)
                    _ = self.model.BaseScore(w_state, '<s>', w_next_state)
                    w_state = w_next_state
                    w_next_state = kenlm.State()
                
                if word_token in self.model:
                    wlm_score = self.model.BaseScore(w_state, word_token, w_next_state)
                        
                if self.hotwords is not None:
                    if word_token in self.hotwords:
                        scored_hw = True
                        wlm_score =  wlm_score + self.hotword_weight
                        
                # OOV
                if abs(wlm_score)==0.0:
                    _ = self.model.BaseScore(w_state, '<unk>', w_next_state)
                    wlm_score = self.oov_offset
                    wlm_score = self.scale_unk*wlm_score
        
        else:
            c_next_state = kenlm.State()
            
            if c_state is None:
                c_state = kenlm.State()
                
                self.char_model.BeginSentenceWrite(c_state)
                _ = self.char_model.BaseScore(c_state, '<s>', c_next_state)
                c_state = c_next_state
                c_next_state = kenlm.State()
                
            # check for OOV
            if char_token not in self.char_model:
                _ = self.char_model.BaseScore(c_state, '<unk>', c_next_state)
                clm_score = self.oov_offset
            else:
                clm_score = self.char_model.BaseScore(c_state, char_token, c_next_state)
        
            if self.hotwords is not None:
                if len(word_token) >= 1:
                    if self.hotwords_tries.has_node(word_token+char_token):
                        min_len = len(next(self.hotwords_tries.iterkeys(word_token, shallow=True)))
                        clm_score = clm_score + self.hotword_weight * len(char_token)/min_len

            # if self.hotwords is not None:
            #     # if len(word_token) >= 1:
            #     if self.hotwords_tries.has_node(word_token+char_token):
            #         min_len = len(next(self.hotwords_tries.iterkeys(word_token, shallow=True)))
            #         clm_score = clm_score + self.hotword_weight * len(char_token)/min_len

        score = clm_score+wlm_score
        if abs(score)>0.0:
            score = self.alpha*score + self.beta
        return score, [c_next_state, w_next_state]
    
    def set_hotwords(self, hotwords: set):
        if hotwords is not None:
            self.hotwords = set(hotwords)
            for w in set(hotwords):
                self.hotwords_tries[w] = 1
        else:
            self.hotwords_tries = CharTrie()
            self.hotwords = hotwords
    
    def score_sentence(self, sentence: str):
        return self.model.score(sentence)

    @property
    def order(self):
        return self.model.order