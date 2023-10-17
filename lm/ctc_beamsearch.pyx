import re
from libc.math cimport log10 as log
cimport numpy as np
from .lm import MultiLevelLM, WordBasedLM, CharBasedLM
np.import_array()

cdef _prune_history(beams, int lm_order, word_boundary_token):
    """Filter out beams that are the same over max_ngram history.

    Since n-gram language models have a finite history when scoring a new token, we can use that
    fact to prune beams that only differ early on (more than n tokens in the past) and keep only the
    higher scoring ones. Note that this helps speed up the decoding process but comes at the cost of
    some amount of beam diversity. If more than the top beam is used in the output it should
    potentially be disabled.
    """
    # let's keep at least 1 word of history
    cdef double word_clm_score, logit_score, lm_score
    cdef double score
    cdef int i
    cdef int B = len(beams)
    cdef int min_n_history = max(1, lm_order - 1)
    cdef list filtered_beams = []
    seen_hashes = set()

    # for each beam after this, check if we need to add it
    for i in range(B):
        seq, last_char, last_word, last_state, start_frame, frame_list, word_clm_score, logit_score, lm_score, score = beams[i]
        # hash based on history that can still affect lm scoring going forward
        # seq = normalize_whitespace(seq)
        hash_idx = (tuple(normalize_whitespace(seq, word_boundary_token).split(word_boundary_token)[-min_n_history:]), last_word, last_char)

        if hash_idx not in seen_hashes:
            filtered_beams.append(
                (
                    seq,
                    last_char,
                    last_word,
                    last_state,
                    start_frame,
                    frame_list,
                    word_clm_score,
                    logit_score,
                    lm_score,
                    score,
                )
            )
            seen_hashes.add(hash_idx)
    return filtered_beams

cdef normalize_whitespace(data, word_boundary_token):
    return re.sub(word_boundary_token+"{2,}", word_boundary_token, data.strip())

cdef get_prune_log_prob(float[:] prob_step, double cutoff_prob, double cutoff_top_n):
    """
    Pruning possible output when cumulative probability has value more than 'cutoff_prob'
    """
    cdef list prob_idx = []
    cdef int cutoff_len = len(prob_step)
    cdef int i
    cdef double cum_prob
    cdef list log_prob_idx = []
    
    for i in range(len(prob_step)):
        prob_idx.append((i, prob_step[i]))
        
    if cutoff_prob < 1.0 or cutoff_top_n < cutoff_len:

        prob_idx.sort(key=lambda t: t[1], reverse=True)
        if cutoff_prob < 1.0:
            cum_prob = 0.0
            cutoff_len = 0
            for i in range(len(prob_idx)):
                cum_prob += prob_idx[i][1]
                cutoff_len += 1
                if (cum_prob >= cutoff_prob or cutoff_len >= cutoff_top_n):
                    break

    for i in range(cutoff_len):
        log_prob_idx.append((
        prob_idx[i][0], log(prob_idx[i][1] + 1e-30)))

    return log_prob_idx
    
"""
def ctc_beamsearch(float[:,:] &logits, vocabulary, lm_scorer=None, word_boundary_token=' ',
                   int beamwidth=64, double cutoff_prob=0.95, int cutoff_top=40 ):
    
    cdef list beams, all_candidates, frame_list, new_frame_list
    
    cdef double log_cutoff_prob
    
    cdef int i, idx, b, start_frame, latest_frame
    cdef int N = len(logits)
    cdef int B

    cdef double word_clm_score
    cdef double score
    cdef double log_prob
    cdef double lm_score, _lm_score
    cdef double min_cutoff
    cdef int min_n_history = 1
    
    cdef float[:] probs
    
    latest_frame = 0
    
    # text, char, word, state, start_frame, frame_list, word_clm_score, logit score, lm score, score
    beams = [
        ["", "", "", None, 0, [], 0, 0, 0, 0]
    ]
    
    if lm_scorer is not None:
        if not isinstance(lm_scorer, CharBasedLM):
            min_n_history = lm_scorer.order - 1
        
        if isinstance(lm_scorer, MultiLevelLM):
            beams = [
                        ["", "", "", [None, None], 0, [], 0, 0, 0, 0]
                    ]
    _vocabulary = vocabulary + ['']

    start_expanding_ = False
    
    # loop over time step
    for i in range(N):
        probs = logits[i]

        all_candidates = []
        # At the start of the decoding process, we delay beam expansion so that
        # timings on the first letters is not incorrect. As soon as we see a
        # timestep with blank probability lower than 0.999, we start expanding
        # beams.

        if (probs[-1] < 0.999):
            start_expanding_ = True

        if not start_expanding_:
            continue

            
        trimed_probs = set(get_prune_log_prob(probs, cutoff_prob, cutoff_top))

        if lm_scorer is None:
            min_cutoff = beams[-1][-1] + log(probs[-1])
        else:
            min_cutoff = beams[-1][-1] + log(probs[-1]) - lm_scorer.beta
        
        # loop over characters prob
        for idx, log_prob in trimed_probs:
            char = _vocabulary[idx]

            B = len(beams)
            for b in range(B):

                _lm_score = 0

                seq, last_char, last_word, last_state, start_frame, frame_list, word_clm_score, logit_score, lm_score, score = beams[b]
                
                if log_prob + score < min_cutoff:
                    continue
                
                if char == "" or last_char == char:
                    candidate = [seq, 
                                char, 
                                last_word, 
                                last_state, 
                                start_frame,
                                frame_list,
                                word_clm_score,
                                logit_score+log_prob,
                                lm_score,
                                score+log_prob]
                else:

                    if lm_scorer is not None:  
                        _lm_score, last_state = lm_scorer.score_state([char, last_word], 
                                                                      last_state)
                        
                    # score word level
                    if char[0] == word_boundary_token and last_word !='':
                        new_frame_list = frame_list.copy()
                        new_frame_list.append([start_frame, latest_frame])

                        # if lm_scorer is not None:
                        if lm_scorer is not None and (isinstance(lm_scorer, MultiLevelLM)):

                            candidate = [seq+char, 
                                        char, 
                                        "", 
                                        last_state, 
                                        -1,
                                        new_frame_list,
                                        0.0,
                                        logit_score+log_prob,
                                        lm_score+_lm_score-word_clm_score,
                                        score+log_prob+_lm_score-word_clm_score]

                        else:
                            candidate = [seq+char, 
                                        char, 
                                        "", 
                                        last_state, 
                                        -1,
                                        new_frame_list,
                                        0.0,
                                        logit_score+log_prob,
                                        lm_score+_lm_score,
                                        score+log_prob+_lm_score]
                  
                    else: # score char level
                        latest_frame = i
                        if start_frame < 0:
                            start_frame = i
                        candidate = [seq+char, 
                                    char, 
                                    last_word+char, 
                                    last_state, 
                                    start_frame,
                                    frame_list,
                                    word_clm_score+_lm_score,
                                    logit_score+log_prob,
                                    lm_score+_lm_score,
                                    score+log_prob+_lm_score]

                all_candidates.append(candidate)

        all_candidates.sort(key=lambda t: t[-1], reverse=True)
        beams = all_candidates[:beamwidth]
        if lm_scorer is not None:
            beams = _prune_history(beams, int(lm_scorer.model.order) if lm_scorer is not None else 1, word_boundary_token)
        else:
            beams = _prune_history(beams, 1, word_boundary_token)

    # final score
    # keep only text, frame and score for last output
    B = len(beams)
    for b in range(B):
        seq, last_char, last_word, last_state, start_frame, frame_list, word_clm_score, logit_score, lm_score, score = beams[b]
        
        seq = normalize_whitespace(seq, word_boundary_token)
        
        _lm_score = 0
        if last_word != '':
            
            new_frame_list = frame_list.copy()
            new_frame_list.append([latest_frame, i])
            
            if lm_scorer is not None:
                _lm_score, last_state = lm_scorer.score_state([word_boundary_token, last_word],
                                                              last_state)
                _lm_score = _lm_score - word_clm_score

                
                end_lm_score, last_state = lm_scorer.score_state(['</s>', '</s>'], last_state)
                _lm_score = _lm_score + end_lm_score
            
            beams[b] = [
                seq, 
                new_frame_list,
                score + _lm_score
            ]
        else:
            beams[b] = [
                seq,
                frame_list,
                score
            ]

    beams.sort(key=lambda t: t[-1], reverse=True)
    
    return beams
"""

def ctc_beamsearch(float[:,:] &logits, vocabulary, lm_scorer=None, word_boundary_token=' ',
                   int beamwidth=64, double cutoff_prob=0.95, int cutoff_top=40 ):
    
    cdef list beams, all_candidates, frame_list, new_frame_list, frame_num
    
    cdef double log_cutoff_prob
    
    cdef int i, idx, b, start_frame, latest_frame
    cdef int N = len(logits)
    cdef int B

    cdef double word_clm_score
    cdef double score
    cdef double log_prob
    cdef double lm_score, _lm_score
    cdef double min_cutoff
    cdef int min_n_history = 1
    
    cdef float[:] probs
    
    latest_frame = 0
    
    # text, char, word, state, start_frame, frame_list, word_clm_score, logit score, lm score, score
    beams = [
        ["", "", "", None, [], [], 0, 0, 0, 0]
    ]
    
    if lm_scorer is not None:
        if not isinstance(lm_scorer, CharBasedLM):
            min_n_history = lm_scorer.order - 1
        
        if isinstance(lm_scorer, MultiLevelLM):
            beams = [
                        ["", "", "", [None, None], [], [], 0, 0, 0, 0]
                    ]
    _vocabulary = vocabulary + ['']

    start_expanding_ = False
    
    # loop over time step
    for i in range(N):
        probs = logits[i]

        all_candidates = []
        # At the start of the decoding process, we delay beam expansion so that
        # timings on the first letters is not incorrect. As soon as we see a
        # timestep with blank probability lower than 0.999, we start expanding
        # beams.

        if (probs[-1] < 0.999):
            start_expanding_ = True

        if not start_expanding_:
            continue

            
        trimed_probs = set(get_prune_log_prob(probs, cutoff_prob, cutoff_top))

        if lm_scorer is None:
            min_cutoff = beams[-1][-1] + log(probs[-1])
        else:
            min_cutoff = beams[-1][-1] + log(probs[-1]) - lm_scorer.beta
        
        # loop over characters prob
        for idx, log_prob in trimed_probs:
            char = _vocabulary[idx]

            B = len(beams)
            for b in range(B):

                _lm_score = 0

                seq, last_char, last_word, last_state, frame_num, frame_list, word_clm_score, logit_score, lm_score, score = beams[b]

                if log_prob + score < min_cutoff:
                    continue
                
                if char == "" or last_char == char:
                    candidate = [seq, 
                                char, 
                                last_word, 
                                last_state, 
                                frame_num,
                                frame_list,
                                word_clm_score,
                                logit_score+log_prob,
                                lm_score,
                                score+log_prob]
                else:

                    if lm_scorer is not None:  
                        _lm_score, last_state = lm_scorer.score_state([char, last_word], 
                                                                      last_state)
                        
                    # score word level
                    if char[0] == word_boundary_token and last_word !='':
                        new_frame_list = frame_list.copy()
                        new_frame_list.append(frame_num)

                        # if lm_scorer is not None:
                        if lm_scorer is not None and (isinstance(lm_scorer, MultiLevelLM)):

                            candidate = [seq+char, 
                                        char, 
                                        "", 
                                        last_state, 
                                        [],
                                        new_frame_list,
                                        0.0,
                                        logit_score+log_prob,
                                        lm_score+_lm_score-word_clm_score,
                                        score+log_prob+_lm_score-word_clm_score]

                        else:
                            candidate = [seq+char, 
                                        char, 
                                        "", 
                                        last_state, 
                                        frame_num+[i],
                                        new_frame_list,
                                        0.0,
                                        logit_score+log_prob,
                                        lm_score+_lm_score,
                                        score+log_prob+_lm_score]
                  
                    else: # score char level
                        # latest_frame = i
                        # if start_frame < 0:
                        #     start_frame = i
                        candidate = [seq+char, 
                                    char, 
                                    last_word+char, 
                                    last_state, 
                                    frame_num+[i],
                                    frame_list,
                                    word_clm_score+_lm_score,
                                    logit_score+log_prob,
                                    lm_score+_lm_score,
                                    score+log_prob+_lm_score]

                all_candidates.append(candidate)

        all_candidates.sort(key=lambda t: t[-1], reverse=True)
        beams = all_candidates[:beamwidth]
        if lm_scorer is not None:
            beams = _prune_history(beams, int(lm_scorer.model.order) if lm_scorer is not None else 1, word_boundary_token)
        else:
            beams = _prune_history(beams, 1, word_boundary_token)

    # final score
    # keep only text, frame and score for last output
    B = len(beams)
    for b in range(B):
        seq, last_char, last_word, last_state, frame_num, frame_list, word_clm_score, logit_score, lm_score, score = beams[b]
        
        seq = normalize_whitespace(seq, word_boundary_token)
        
        _lm_score = 0
        if last_word != '':
            
            new_frame_list = frame_list.copy()
            new_frame_list.append(frame_num)
            
            if lm_scorer is not None:
                _lm_score, last_state = lm_scorer.score_state([word_boundary_token, last_word],
                                                              last_state)
                _lm_score = _lm_score - word_clm_score

                
                end_lm_score, last_state = lm_scorer.score_state(['</s>', '</s>'], last_state)
                _lm_score = _lm_score + end_lm_score
            
            beams[b] = [
                seq, 
                new_frame_list,
                score + _lm_score
            ]
        else:
            beams[b] = [
                seq,
                frame_list,
                score
            ]

    beams.sort(key=lambda t: t[-1], reverse=True)
    
    return beams