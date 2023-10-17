import numpy as np
import onnxruntime as ort

from scipy.special import log_softmax

DEFAULT_BLANK_PROB = 0.9999

class WordConfidenceEstimator:
    """
        Estimate word confidence
        most of code implement by burin naowarat
    """
    def __init__(self, configs) -> None:
        self.sess = ort.InferenceSession(configs.model_path)

        self.blank_index = configs.blank_index
        self.space_index = configs.space_index

    def infer(self, pred, pred_prob):
        features, features_len = self.aggregation(pred, pred_prob, 
                                                  maxlen=len(np.where(pred==self.space_index)[1])+1, 
                                                  by=self.space_index, agg='mean')      

        confidence = self.sess.run(None, {'features': features})[0]
        return confidence


    def merge_repeated_batch(self, logits, blank_index=None, skip_blank=False, use_logits=True, agg='max'):
        # x : (B, T)
        # input logits : batch_size, output_len, num_classes
        # return h : index of logits, which is has the highest probs for merged alphabets

        B, T, C =logits.shape
        x = logits.argmax(axis=2) # batch_size, output_len
        v = logits.max(axis=2) # batch_size, output_len

        # TODO store index in h_prob for efficiency
        # h_prob = np.full( (B, T), 0, dtype=np.int64) # keep index of logits 
        if use_logits: # logit mode
            default_prob = np.full( (C, ), 0., dtype=np.float32)
        else: # logprob mode
            default_prob = np.full( (C, ), np.log((1-DEFAULT_BLANK_PROB)/(C-1)), dtype=np.float32)
            default_prob[blank_index] = np.log(DEFAULT_BLANK_PROB)
        h_prob = np.full( (B, T, C), default_prob, dtype=np.float32) # target logprobs

        h = np.full( (B, T), 0, dtype=np.int64) # alphabet list
        cur_position = np.full((B,), 0, dtype=np.int64) # position on h

        cur_max_v = np.full( (B, ), 0, dtype=np.int32) # position of the logits for the latest alphabet        
        if not skip_blank:
            not_blank = x[:, 0] == x[:, 0] 
        

        for t in range(0, h.shape[1]):
            if skip_blank: not_blank = x[:,t] != blank_index
            if t > 0: change = (x[:, t] != x[:, t-1]) & not_blank
            else: change = not_blank

            # Use max prob for for merged postition
            if agg == 'max':
                update = (v[np.arange(B), cur_max_v] < v[:, t]) & not_blank # (B, T)
            elif agg == 'min':
                update = (v[np.arange(B), cur_max_v] > v[:, t]) & not_blank # (B, T)
            else:
                raise ValueError("agg {} is not implemented")
            cur_max_v[ change | update ] = t
            
            h_prob[np.arange(B), cur_position] = logits[np.arange(B), cur_max_v]
            h[np.arange(B), cur_position] = x[:, t]

            cur_position[change] += 1
        
        h_len = cur_position

        return h, h_prob, h_len

    def aggregation(self, pred, pred_logits, maxlen, by=0, use_logits=True, agg='mean'):
        # pred;  (B, T)
        # pred_logits; (B, T, C)
        # maxlen : int ; the biggest number of word across samples inside the batch
        # by; int := blank_index ; index used for spliting
        # agg; mean, last, max    
        # return featuress (B, maxlen, C)
        
        B, T, C = pred_logits.shape

        cur_position = np.full((B,), 0, dtype=np.int64) # indicates position in aggregated_logits
        token_counter = np.full( (B,1), 1, dtype=np.float32) # for mean


        candidate_onehot = np.full( (B, C), 0., dtype=np.float32)
        aggregated_onehots = np.full( (B, maxlen, C), 0., dtype=np.float32)

        if use_logits: # logit mode
            if agg == 'geomean':
                default_prob = np.full( (C, ), 1., dtype=np.float32)
            elif agg == 'min':
                default_prob = np.full( (C, ), np.inf, dtype=np.float32)
            else:
                default_prob = np.full( (C, ), 0., dtype=np.float32)
        else: # logprob mode
            raise NotImplementedError()
            # default_prob = np.full( (C, ), np.log((1-self.default_blank_prob)/(C-1)), dtype=np.float32)
            # default_prob[self.blank_index] = np.log(self.default_blank_prob)
            
        candidate = np.full( (B, C), default_prob, dtype=np.float32)
        aggregated_logits = np.full( (B, maxlen, C), 0., dtype=np.float32)


        if agg == 'min':
            x = pred_logits.argmax(axis=2) # batch_size, output_len
            v = pred_logits.max(axis=2) # batch_size, output_len

        if agg == 'geomean':
            # turn into log softmax
            pred_logits = np.log(pred_logits)

        if agg != None:
            for t in range(1, pred.shape[1]):
                found_boundary = pred[:, t] == by
                # print(candidate.shape, found_boundary, token_counter.shape)

                # update states
                if agg in ['mean', 'geomean']:
                    candidate[found_boundary, :] /= token_counter[found_boundary]
                    candidate[~found_boundary] += pred_logits[~found_boundary, t, :]
                elif agg == 'last':
                    candidate[~found_boundary] = pred_logits[~found_boundary, t, :]
                elif agg == 'min':
                    update = candidate.max(axis=-1) > v[:, t]
                    # candidate is the up to date lowest logits within the current word boundary
                    candidate[~found_boundary & update, :] = pred_logits[~found_boundary&update, t, :]
                else:
                    raise NotImplementedError("agg:={} have not implemented.".format(agg))

                # update results
                # print(cur_position, t)
                aggregated_logits[np.arange(B), cur_position] = candidate
                aggregated_onehots[found_boundary, cur_position[found_boundary]] = candidate_onehot[found_boundary]

                # reset states
                candidate[found_boundary] = default_prob
                candidate_onehot[~found_boundary, pred[~found_boundary, t]] += 1.
                candidate_onehot[found_boundary, pred[found_boundary, t]] = 0.

                # update iterator
                cur_position[found_boundary] += 1
                # if cur_position == maxlen-1:
                #     break

            # Finalized the lastest alphabets
            if agg in ['mean', 'geomean']:
                candidate[~found_boundary, :] /= token_counter[~found_boundary]
                aggregated_logits[np.arange(B), cur_position] = candidate
            
            if agg == 'geomean':
                # convert back to probability scales
                aggregated_logits = np.exp(aggregated_logits)

            if agg == 'min':
                aggregated_logits[np.arange(B), cur_position] = candidate            
                aggregated_logits[np.isinf(aggregated_logits)] = 0.

            aggregated_onehots[~found_boundary, cur_position[~found_boundary]] = candidate_onehot[~found_boundary] 

        agg_len = np.array(cur_position + 1) # surplus the last word into length

        features = aggregated_logits
        features = np.concatenate([features, aggregated_onehots], axis=-1) #(B, maxlen, 2C)
        features = np.concatenate([features, log_softmax(aggregated_logits, axis=-1)], axis=-1)
        
        # (B, maxlen, dim), (B, )
        return features, agg_len