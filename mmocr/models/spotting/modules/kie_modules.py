#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/4/2 22:22
# @Author : WeiHua

# todo: prepare CRF module
"""
Ref:
[1] https://towardsdatascience.com/conditional-random-fields-explained-e5b8256da776
[2] https://zhuanlan.zhihu.com/p/70067113
[3] https://zhuanlan.zhihu.com/p/104562658
[4] https://zhuanlan.zhihu.com/p/38119194
[5] https://zhuanlan.zhihu.com/p/148813079

"""

# inherited from PICK: https://github.com/wenwenyu/PICK-pytorch

# from typing import List, Tuple, Dict
from typing import *

import torch
import torch.nn as nn

# from allennlp.common.checks import ConfigurationError
# import allennlp.nn.util as util

from torchcrf import CRF

'''
Copy-paste from allennlp.modules.conditional_random_field 
with modifications:
    * viterbi_tags output the best path insted of top k paths
'''


def allowed_transitions(constraint_type: str, labels: Dict[int, str]) -> List[Tuple[int, int]]:
    """
    Given labels and a constraint type, returns the allowed transitions. It will
    additionally include transitions for the start and end states, which are used
    by the conditional random field.

    Parameters
    ----------
    constraint_type : ``str``, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    labels : ``Dict[int, str]``, required
        A mapping {label_id -> label}. Most commonly this would be the value from
        Vocabulary.get_index_to_token_vocabulary()

    Returns
    -------
    ``List[Tuple[int, int]]``
        The allowed transitions (from_label_id, to_label_id).
    """
    num_labels = len(labels)
    start_tag = num_labels
    end_tag = num_labels + 1
    labels_with_boundaries = list(labels.items()) + [(start_tag, "START"), (end_tag, "END")]

    allowed = []
    for from_label_index, from_label in labels_with_boundaries:
        if from_label in ("START", "END"):
            from_tag = from_label
            from_entity = ""
        else:
            from_tag = from_label[0]
            from_entity = from_label[1:]
        for to_label_index, to_label in labels_with_boundaries:
            if to_label in ("START", "END"):
                to_tag = to_label
                to_entity = ""
            else:
                to_tag = to_label[0]
                to_entity = to_label[1:]
            if is_transition_allowed(constraint_type, from_tag, from_entity,
                                     to_tag, to_entity):
                allowed.append((from_label_index, to_label_index))
    return allowed


def is_transition_allowed(constraint_type: str,
                          from_tag: str,
                          from_entity: str,
                          to_tag: str,
                          to_entity: str):
    """
    Given a constraint type and strings ``from_tag`` and ``to_tag`` that
    represent the origin and destination of the transition, return whether
    the transition is allowed under the given constraint type.

    Parameters
    ----------
    constraint_type : ``str``, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    from_tag : ``str``, required
        The tag that the transition originates from. For example, if the
        label is ``I-PER``, the ``from_tag`` is ``I``.
    from_entity: ``str``, required
        The entity corresponding to the ``from_tag``. For example, if the
        label is ``I-PER``, the ``from_entity`` is ``PER``.
    to_tag : ``str``, required
        The tag that the transition leads to. For example, if the
        label is ``I-PER``, the ``to_tag`` is ``I``.
    to_entity: ``str``, required
        The entity corresponding to the ``to_tag``. For example, if the
        label is ``I-PER``, the ``to_entity`` is ``PER``.

    Returns
    -------
    ``bool``
        Whether the transition is allowed under the given ``constraint_type``.
    """
    # pylint: disable=too-many-return-statements
    if to_tag == "START" or from_tag == "END":
        # Cannot transition into START or from END
        return False

    if constraint_type == "BIOUL":
        if from_tag == "START":
            return to_tag in ('O', 'B', 'U')
        if to_tag == "END":
            return from_tag in ('O', 'L', 'U')
        return any([
            # O can transition to O, B-* or U-*
            # L-x can transition to O, B-*, or U-*
            # U-x can transition to O, B-*, or U-*
            from_tag in ('O', 'L', 'U') and to_tag in ('O', 'B', 'U'),
            # B-x can only transition to I-x or L-x
            # I-x can only transition to I-x or L-x
            from_tag in ('B', 'I') and to_tag in ('I', 'L') and from_entity == to_entity
        ])
    elif constraint_type == "BIO":
        if from_tag == "START":
            return to_tag in ('O', 'B')
        if to_tag == "END":
            return from_tag in ('O', 'B', 'I')
        return any([
            # Can always transition to O or B-x
            to_tag in ('O', 'B'),
            # Can only transition to I-x from B-x or I-x
            to_tag == 'I' and from_tag in ('B', 'I') and from_entity == to_entity
        ])
    elif constraint_type == "IOB1":
        if from_tag == "START":
            return to_tag in ('O', 'I')
        if to_tag == "END":
            return from_tag in ('O', 'B', 'I')
        return any([
            # Can always transition to O or I-x
            to_tag in ('O', 'I'),
            # Can only transition to B-x from B-x or I-x, where
            # x is the same tag.
            to_tag == 'B' and from_tag in ('B', 'I') and from_entity == to_entity
        ])
    elif constraint_type == "BMES":
        if from_tag == "START":
            return to_tag in ('B', 'S')
        if to_tag == "END":
            return from_tag in ('E', 'S')
        return any([
            # Can only transition to B or S from E or S.
            to_tag in ('B', 'S') and from_tag in ('E', 'S'),
            # Can only transition to M-x from B-x, where
            # x is the same tag.
            to_tag == 'M' and from_tag in ('B', 'M') and from_entity == to_entity,
            # Can only transition to E-x from B-x or M-x, where
            # x is the same tag.
            to_tag == 'E' and from_tag in ('B', 'M') and from_entity == to_entity,
        ])
    else:
        raise ConfigurationError(f"Unknown constraint type: {constraint_type}")


class ConditionalRandomField(torch.nn.Module):
    """
    This module uses the "forward-backward" algorithm to compute
    the log-likelihood of its inputs assuming a conditional random field model.

    See, e.g. http://www.cs.columbia.edu/~mcollins/fb.pdf

    Parameters
    ----------
    num_tags : int, required
        The number of tags.
    constraints : List[Tuple[int, int]], optional (default: None)
        An optional list of allowed transitions (from_tag_id, to_tag_id).
        These are applied to ``viterbi_tags()`` but do not affect ``forward()``.
        These should be derived from `allowed_transitions` so that the
        start and end transitions are handled correctly for your tag type.
    include_start_end_transitions : bool, optional (default: True)
        Whether to include the start and end transition parameters.
    """

    def __init__(self,
                 num_tags: int,
                 constraints: List[Tuple[int, int]] = None,
                 include_start_end_transitions: bool = True) -> None:
        super().__init__()
        self.num_tags = num_tags

        # transitions[i, j] is the logit for transitioning from state i to state j.
        self.transitions = torch.nn.Parameter(torch.Tensor(num_tags, num_tags))

        # _constraint_mask indicates valid transitions (based on supplied constraints).
        # Include special start of sequence (num_tags + 1) and end of sequence tags (num_tags + 2)
        if constraints is None:
            # All transitions are valid.
            constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(1.)
        else:
            constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(0.)
            for i, j in constraints:
                constraint_mask[i, j] = 1.

        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        # Also need logits for transitioning from "start" state and to "end" state.
        self.include_start_end_transitions = include_start_end_transitions
        if include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        """
        batch_size, sequence_length, num_tags = logits.size()

        # Transpose batch size and sequence dimensions
        mask = mask.float().transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]

        # For each i we compute logits for the transitions from timestep i-1 to timestep i.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are
        # (instance, current_tag, next_tag)
        for i in range(1, sequence_length):
            # The emit scores are for time i ("next_tag") so we broadcast along the current_tag axis.
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            # Transition scores are (current_tag, next_tag) so we broadcast along the instance axis.
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            # Alpha is for the current_tag, so we broadcast along the next_tag axis.
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # Add all the scores together and logexp over the current_tag axis
            inner = broadcast_alpha + emit_scores + transition_scores

            # In valid positions (mask == 1) we want to take the logsumexp over the current_tag dimension
            # of ``inner``. Otherwise (mask == 0) we want to retain the previous alpha.
            alpha = (util.logsumexp(inner, 1) * mask[i].view(batch_size, 1) +
                     alpha * (1 - mask[i]).view(batch_size, 1))

        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return util.logsumexp(stops)

    def _joint_likelihood(self,
                          logits: torch.Tensor,
                          tags: torch.Tensor,
                          mask: torch.LongTensor) -> torch.Tensor:
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
        """
        batch_size, sequence_length, _ = logits.data.shape

        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        # Start with the transition scores from start_tag to the first tag in each input
        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0

        # Add up the scores for the observed transitions and all the inputs but the last
        for i in range(sequence_length - 1):
            # Each is shape (batch_size,)
            current_tag, next_tag = tags[i], tags[i + 1]

            # The scores for transitioning from current_tag to next_tag
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]

            # The score for using current_tag
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)

            # Include transition score if next element is unmasked,
            # input_score if this element is unmasked.
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]

        # Transition from last state to "stop" state. To start with, we need to find the last tag
        # for each instance.
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)

        # Compute score of transitioning to `stop_tag` from each "last tag".
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        # Add the last input if it's not masked.
        last_inputs = logits[-1]  # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()  # (batch_size,)

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def forward(self,
                inputs: torch.Tensor,
                tags: torch.Tensor,
                mask: torch.ByteTensor = None, input_batch_first=False, keepdim=False) -> torch.Tensor:
        """
        Computes the log likelihood. inputs, tags, mask are assumed to be batch first
        """
        # convert to batch_first
        if not input_batch_first:
            inputs = inputs.transpose(0, 1).contiguous()
            tags = tags.transpose(0, 1).contiguous()
            if mask is not None:
                mask = mask.transpose(0, 1).contiguous()

        # pylint: disable=arguments-differ
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.long)

        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)

        if keepdim:
            return log_numerator - log_denominator
        else:
            return torch.sum(log_numerator - log_denominator)

    def viterbi_tags(self,
                     logits: torch.Tensor,
                     mask: torch.Tensor, logits_batch_first=False) -> List[Tuple[List[int], float]]:
        """
        Uses viterbi algorithm to find most likely tags for the given inputs.
        If constraints are applied, disallows all other transitions.
        """

        if not logits_batch_first:
            logits = logits.transpose(0, 1).contiguous()
            mask = mask.transpose(0, 1).contiguous()

        _, max_seq_length, num_tags = logits.size()

        # Get the tensors out of the variables
        logits, mask = logits.data, mask.data

        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.)

        # Apply transition constraints
        constrained_transitions = (
                self.transitions * self._constraint_mask[:num_tags, :num_tags] +
                -10000.0 * (1 - self._constraint_mask[:num_tags, :num_tags])
        )
        transitions[:num_tags, :num_tags] = constrained_transitions.data

        if self.include_start_end_transitions:
            transitions[start_tag, :num_tags] = (
                    self.start_transitions.detach() * self._constraint_mask[start_tag, :num_tags].data +
                    -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            )
            transitions[:num_tags, end_tag] = (
                    self.end_transitions.detach() * self._constraint_mask[:num_tags, end_tag].data +
                    -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())
            )
        else:
            transitions[start_tag, :num_tags] = (-10000.0 *
                                                 (1 - self._constraint_mask[start_tag, :num_tags].detach()))
            transitions[:num_tags, end_tag] = -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())

        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)

        for prediction, prediction_mask in zip(logits, mask):
            sequence_length = torch.sum(prediction_mask)

            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.
            # At steps 1, ..., sequence_length we just use the incoming prediction
            tag_sequence[1:(sequence_length + 1), :num_tags] = prediction[:sequence_length]
            # And at the last timestep we must have the END_TAG
            tag_sequence[sequence_length + 1, end_tag] = 0.

            # We pass the tags and the transitions to ``viterbi_decode``.
            viterbi_path, viterbi_score = util.viterbi_decode(tag_sequence[:(sequence_length + 2)], transitions)
            # Get rid of START and END sentinels and append.
            viterbi_path = viterbi_path[1:-1]
            best_paths.append((viterbi_path, viterbi_score.item()))

        return best_paths


class MLPLayer(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: Optional[int] = None,
                 hidden_dims: Optional[List[int]] = None,
                 layer_norm: bool = False,
                 dropout: Optional[float] = 0.0,
                 activation: Optional[str] = 'relu'):
        '''
        transform output of LSTM layer to logits, as input of crf layers
        :param in_dim:
        :param out_dim:
        :param hidden_dims:
        :param layer_norm:
        :param dropout:
        :param activation:
        '''
        super().__init__()
        layers = []
        activation_layer = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU
        }

        if hidden_dims:
            for dim in hidden_dims:
                layers.append(nn.Linear(in_dim, dim))
                layers.append(activation_layer.get(activation, nn.Identity()))
                Warning(
                    'Activation function {} is not supported, and replace with Identity layer.'.format(activation))
                if layer_norm:
                    layers.append(nn.LayerNorm(dim))
                if dropout:
                    layers.append(nn.Dropout(dropout))
                in_dim = dim

        if not out_dim:
            layers.append(nn.Identity())
        else:
            layers.append(nn.Linear(in_dim, out_dim))

        self.mlp = nn.Sequential(*layers)
        self.out_dim = out_dim if out_dim else hidden_dims[-1]

    def forward(self, *input: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat(input, 1))


class BiLSTMLayer(nn.Module):

    def __init__(self, lstm_kwargs, mlp_kwargs, pad_val, apply_norm=False):
        super().__init__()
        self.lstm = nn.LSTM(**lstm_kwargs)
        self.mlp = MLPLayer(**mlp_kwargs)
        self.apply_norm = apply_norm
        if apply_norm:
            self.in_norm = nn.LayerNorm(lstm_kwargs.get('input_size'))
            self.out_norm = nn.LayerNorm(mlp_kwargs.get('out_dim'))
        self.pad_val = pad_val

    @staticmethod
    def sort_tensor(x: torch.Tensor, length: torch.Tensor, h_0: torch.Tensor = None, c_0: torch.Tensor = None):
        sorted_lenght, sorted_order = torch.sort(length, descending=True)
        _, invert_order = sorted_order.sort(0, descending=False)
        if h_0 is not None:
            h_0 = h_0[:, sorted_order, :]
        if c_0 is not None:
            c_0 = c_0[:, sorted_order, :]
        return x[sorted_order], sorted_lenght, invert_order, h_0, c_0

    def forward(self, x_seq: torch.Tensor,
                lenghts: torch.Tensor,
                initial: Tuple[torch.Tensor, torch.Tensor]):
        '''

        :param x_seq: (B, N*T, D)
        :param lenghts: (B,)
        :param initial: (num_layers * directions, batch, D)
        :return: (B, N*T, out_dim)
        '''
        if self.apply_norm:
            x_seq = self.in_norm(x_seq)
        # B*N, T, hidden_size
        x_seq, sorted_lengths, invert_order, h_0, c_0 = self.sort_tensor(x_seq, lenghts, initial[0], initial[0])
        # length.cpu() is adapted torch1.7
        packed_x = nn.utils.rnn.pack_padded_sequence(x_seq, lengths=sorted_lengths.cpu(), batch_first=True)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True,
                                                     padding_value=self.pad_val)
        # total_length=documents.MAX_BOXES_NUM * documents.MAX_TRANSCRIPT_LEN
        output = output[invert_order]
        logits = self.mlp(output)
        # (B, N*T, out_dim)
        if self.apply_norm:
            logits = self.out_norm(logits)
        return logits


class UnionLayer(nn.Module):

    def __init__(self, debug_mode=False, sort_by_length=False):
        super().__init__()
        self.debug_mode = debug_mode
        self.sort_by_length = sort_by_length

    def forward(self, kie_logits_, texts_, end_val, tags_=None, tag_pad=-1,
                sorted_idx=None):
        '''
        Merge non-padding transcripts to document-level. The input order should be reading order by default.
        'SOS' of these input should be removed before.
        :param kie_logits_: the output of KIE's transformer layer, (B, N, L, D)
        :param texts: (B, N, L)
        :param tags: IBO label for every segments of documents, (B, N, T)
        :param end_val: end value, which indicates the end of the line
        :param sorted_idx: list of lists, which indicates the origin order of boxes
        :return:
                new_kie_logits, (B, max_doc_seq_len, D)
                new_mask, (B, max_doc_seq_len)
                new_tags, (B, max_doc_seq_len)
                mask, (B, N, L)
        '''
        B, N, L, C = kie_logits_.shape
        if sorted_idx:
            kie_logits = torch.zeros_like(kie_logits_, device=kie_logits_.device)
            texts = torch.zeros_like(texts_, device=texts_.device)
            if self.training:
                tags = torch.zeros_like(tags_, device=tags_.device)
            for i in range(B):
                gt_order = sorted_idx[i]

                kie_logits[i, :len(gt_order), :, :] = kie_logits_[i, gt_order, :, :]
                kie_logits[i, len(gt_order):, :, :] = kie_logits_[i, len(gt_order):, :, :]

                texts[i, :len(gt_order), :] = texts_[i, gt_order, :]
                texts[i, len(gt_order):, :] = texts_[i, len(gt_order):, :]

                if self.training:
                    tags[i, :len(gt_order), :] = tags_[i, gt_order, :]
                    tags[i, len(gt_order):, :] = tags_[i, len(gt_order):, :]
        else:
            kie_logits = kie_logits_
            texts = texts_
            tags = tags_

        kie_logits = kie_logits.reshape(B, N*L, C)
        # generate non-pad mask, where True indicates non-pad
        mask = torch.zeros((B, N, L), device=kie_logits.device).bool()
        doc_lengths = []
        for i in range(B):
            doc_len = 0
            for j in range(N):
                end_poses = torch.nonzero(texts[i, j] == end_val)
                if end_poses.shape[0] > 0:
                    mask[i, j, :end_poses[0, 0]] = True
                    doc_len += end_poses[0, 0]
            doc_lengths.append(doc_len)
        mask = mask.reshape(B, N*L)

        max_doc_seq_len = max(doc_lengths)
        new_kie_logits = torch.zeros_like(kie_logits, device=kie_logits.device)
        new_mask = torch.zeros((B, N*L), device=kie_logits.device).bool()
        if self.training:
            tags = tags.reshape(B, N*L)
            new_tags = torch.full_like(tags, tag_pad, device=kie_logits.device)
            new_tags = new_tags[:, :max_doc_seq_len]

        if self.debug_mode:
            new_texts = texts.reshape(B, N*L)
            doc_texts = torch.full_like(new_tags, end_val, device=kie_logits.device)

        for i in range(B):
            doc_logits = kie_logits[i]
            doc_mask = mask[i]
            # num_valid, D
            valid_doc_logits = doc_logits[doc_mask]
            # B, N*L, D
            new_kie_logits[i, :doc_lengths[i]] = valid_doc_logits
            # B, N*L
            new_mask[i, :doc_lengths[i]] = True

            if self.training:
                valid_tags = tags[i, doc_mask]
                new_tags[i, :doc_lengths[i]] = valid_tags

            if self.debug_mode:
                valid_texts = new_texts[i, doc_mask]
                doc_texts[i, :doc_lengths[i]] = valid_texts

        # (B, max_doc_seq_len, D)
        new_kie_logits = new_kie_logits[:, :max_doc_seq_len, :]
        # (B, max_doc_seq_len)
        new_mask = new_mask[:, :max_doc_seq_len]

        if self.training:
            if self.debug_mode:
                return new_kie_logits, new_mask, new_tags, mask.reshape(B, N, L), doc_texts
            return new_kie_logits, new_mask, new_tags, mask.reshape(B, N, L)
        else:
            return new_kie_logits, new_mask, mask.reshape(B, N, L)

    def split_out(self, best_paths, mask, sorted_idx=None):
        """
        Split document-level logits into origin instance-level
        Args:
            best_paths: [[tags of one image], ...]
            mask: (B, N, L)
            sorted_idx: list of lists, which indicates the origin order of boxes

        Returns:
            instance-level logits, with shape (B, N, L, C)
        """
        B, N, L = mask.shape
        ins_lvl_tags = torch.zeros((B, N, L), device=mask.device)
        for i in range(B):
            st_idx = 0
            for j in range(N):
                num_valid = len(torch.nonzero(mask[i, j]))
                ins_lvl_tags[i, j, :num_valid] = torch.Tensor(best_paths[i][st_idx: st_idx+num_valid])
                st_idx += num_valid
            assert st_idx == len(best_paths[i]), f"End at {st_idx} while length is {len(best_paths[i])}"
        if sorted_idx:
            for i in range(B):
                rev_sorted_idx = [sorted_idx[i].index(x) for x in range(len(sorted_idx[i]))]
                ins_lvl_tags[i, :len(rev_sorted_idx), :] = ins_lvl_tags[i, rev_sorted_idx, :]

        return ins_lvl_tags

    def split_logits_out(self, logits, mask, sorted_idx=None):
        """
        Split document-level logits into origin instance-level
        Args:
            logits: Tensor, (B, doc_len, C)
            mask: (B, N, L)
            sorted_idx: list of lists, which indicates the origin order of boxes

        Returns:
            instance-level logits, with shape (B*N, L, C)
        """
        B, N, L = mask.shape
        C = logits.shape[-1]
        ins_lvl_logits = torch.zeros((B, N, L, C), device=mask.device)
        for i in range(B):
            st_idx = 0
            for j in range(N):
                num_valid = len(torch.nonzero(mask[i, j]))
                ins_lvl_logits[i, j, :num_valid] = logits[i, st_idx: st_idx + num_valid]
                st_idx += num_valid
        if sorted_idx:
            for i in range(B):
                rev_sorted_idx = [sorted_idx[i].index(x) for x in range(len(sorted_idx[i]))]
                ins_lvl_logits[i, :len(rev_sorted_idx), ...] = ins_lvl_logits[i, rev_sorted_idx, ...]

        return ins_lvl_logits.reshape(B*N, L, C)


class KIEDecoder(nn.Module):
    def __init__(self, ocr_dict, entity_dict, use_crf=False, use_kie_loss=False, ins_lvl_mean=False):
        super(KIEDecoder, self).__init__()
        self.debug_mode = False
        self.use_kie_loss = use_kie_loss
        self.ins_lvl_mean = ins_lvl_mean
        self.ocr_dict = ocr_dict
        self.rev_ocr_dict = dict()
        for key, val in ocr_dict.items():
            self.rev_ocr_dict[val] = key
        self.entity_dict = entity_dict
        self.use_crf = use_crf
        if use_crf:
            # self.crf = ConditionalRandomField(num_tags=len(entity_dict))
            self.crf = CRF(len(entity_dict), batch_first=True)
            self.union_layer = UnionLayer(debug_mode=self.debug_mode)
            self.soft_max_func = nn.Softmax(dim=-1)

    def forward(self, kie_logits, texts=None, tags=None, logits_logger=None,
                sorted_idx=None):
        """
        Decoder of KIE branch
        Args:
            kie_logits: B*N, L, C
            texts: B, N, L
            tags: B, N, L
            logits_logger: dict

        Returns:

        """
        logits_logger['KIE'].append(kie_logits)
        if self.training and self.use_crf:
            if not self.use_kie_loss:
                logits_logger['KIE'].pop()
            B, N, L = texts.shape
            if not self.debug_mode:
                new_kie_logits, new_mask, new_tags, mask = self.union_layer(
                    kie_logits.reshape(B, N, L, -1),
                    texts, self.ocr_dict['<END>'],
                    tags_=tags,
                    tag_pad=self.entity_dict['<PAD>'],
                    sorted_idx=sorted_idx)
            else:
                new_kie_logits, new_mask, new_tags, mask, doc_texts = self.union_layer(
                    kie_logits.reshape(B, N, L, -1),
                    texts, self.ocr_dict['<END>'],
                    tags_=tags,
                    tag_pad=self.entity_dict['<PAD>'],
                    sorted_idx=sorted_idx)
                docs = []
                for i in range(B):
                    cur_doc = ""
                    for j in doc_texts[i]:
                        cur_idx = j.item()
                        if cur_idx == self.ocr_dict['<END>']:
                            break
                        cur_doc += self.rev_ocr_dict[cur_idx]
                    docs.append(cur_doc)
                import ipdb
                ipdb.set_trace()

            # log_likelihood = self.crf(new_kie_logits, new_tags, new_mask,
            #                           input_batch_first=True,
            #                           keepdim=True)
            if self.ins_lvl_mean:
                log_likelihood = self.crf(new_kie_logits, new_tags, mask=new_mask)
                if sorted_idx:
                    total_num_box = 0
                    for idx_set in sorted_idx:
                        total_num_box += len(idx_set)
                    log_likelihood /= total_num_box
                else:
                    raise ValueError(f"sorted_idx is required for calculating the num of boxes")
            else:
                log_likelihood = self.crf(new_kie_logits, new_tags, mask=new_mask,
                                          reduction='mean')

            if 'CRF' not in logits_logger:
                logits_logger['CRF'] = [-log_likelihood.reshape(1)]
            else:
                log_likelihood['CRF'].append(-log_likelihood.reshape(1))

    def crf_decode(self, logits_logger, shape_, sorted_idx):
        if self.training:
            Warning("crf_decode should be inference only")
        if self.crf:
            B, N = shape_
            C = logits_logger['KIE'][-1].shape[-1]
            _, pred_texts = torch.max(self.soft_max_func(logits_logger['REC'][-1]), dim=-1)
            new_kie_logits, new_mask, mask = self.union_layer(
                logits_logger['KIE'][-1].reshape(B, N, -1, C),
                pred_texts.reshape(B, N, -1),
                self.ocr_dict['<END>'],
                sorted_idx=sorted_idx)
            # tmp = self.crf.viterbi_tags(new_kie_logits, new_mask, logits_batch_first=True)
            best_paths = self.crf.decode(new_kie_logits, new_mask)
            ins_lvl_tags = self.union_layer.split_out(best_paths, mask, sorted_idx=sorted_idx)
            if 'CRF' not in logits_logger:
                logits_logger['CRF'] = [ins_lvl_tags]
            else:
                logits_logger['CRF'].append(ins_lvl_tags)

# todo:
#   1. full analysis over current model, and determine how to convert instance-level prediction
#   process to document-level. In a word, focus on layout modeling, context modeling and CRF, with
#   proper design, this could be treated as a contribution.


class KIEDecoderSerial(nn.Module):
    """ KIE decoder for serial structure"""
    def __init__(self, ocr_dict, entity_dict, use_crf=False, use_kie_loss=False, ins_lvl_mean=False,
                 d_model=-1, lstm_args=None, use_both_modal=False):
        super(KIEDecoderSerial, self).__init__()
        self.debug_mode = False
        self.use_kie_loss = use_kie_loss
        self.ins_lvl_mean = ins_lvl_mean
        self.use_both_modal = use_both_modal
        self.ocr_dict = ocr_dict
        self.rev_ocr_dict = dict()
        for key, val in ocr_dict.items():
            self.rev_ocr_dict[val] = key
        self.entity_dict = entity_dict

        bilstm_kwargs = lstm_args.get('bilstm_kwargs', dict())
        mlp_kwargs = lstm_args.get('mlp_kwargs', dict())
        bilstm_kwargs.update(input_size=2*d_model if use_both_modal else d_model)
        if bilstm_kwargs['bidirectional']:
            mlp_kwargs.update(in_dim=2*bilstm_kwargs['hidden_size'])
        else:
            mlp_kwargs.update(in_dim=bilstm_kwargs['hidden_size'])
        mlp_kwargs.update(out_dim=len(self.entity_dict))
        self.bilstm_layer = BiLSTMLayer(bilstm_kwargs, mlp_kwargs,
                                        pad_val=self.ocr_dict['<PAD>'],
                                        apply_norm=lstm_args.get('apply_norm', False))

        self.union_layer = UnionLayer(debug_mode=self.debug_mode)
        self.soft_max_func = nn.Softmax(dim=-1)

        self.use_crf = use_crf
        if use_crf:
            # self.crf = ConditionalRandomField(num_tags=len(entity_dict))
            self.crf = CRF(len(entity_dict), batch_first=True)
        else:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.entity_dict['<PAD>'],
                reduction='mean'
            )

    def forward(self, kie_logits, kie_cls_res, texts=None, tags=None, logits_logger=None,
                sorted_idx=None, rec_logits=None):
        """
        Decoder of KIE branch
        Args:
            kie_logits: B*N, L, C
            texts: B, N, L
            tags: B, N, L
            logits_logger: dict

        Returns:

        """
        logits_logger['KIE'].append(kie_cls_res)
        if self.training:
            if not self.use_kie_loss:
                logits_logger['KIE'].pop()
            B, N, L = texts.shape
            if not self.debug_mode:
                if self.use_both_modal:
                    new_kie_logits, new_mask, new_tags, mask = self.union_layer(
                        torch.cat((kie_logits, rec_logits), dim=-1).reshape(B, N, L, -1),
                        texts, self.ocr_dict['<END>'],
                        tags_=tags,
                        tag_pad=self.entity_dict['<PAD>'],
                        sorted_idx=sorted_idx)
                else:
                    new_kie_logits, new_mask, new_tags, mask = self.union_layer(
                        kie_logits.reshape(B, N, L, -1),
                        texts, self.ocr_dict['<END>'],
                        tags_=tags,
                        tag_pad=self.entity_dict['<PAD>'],
                        sorted_idx=sorted_idx)
            else:
                raise NotImplementedError
                new_kie_logits, new_mask, new_tags, mask, doc_texts = self.union_layer(
                    kie_logits.reshape(B, N, L, -1),
                    texts, self.ocr_dict['<END>'],
                    tags_=tags,
                    tag_pad=self.entity_dict['<PAD>'],
                    sorted_idx=sorted_idx)
                docs = []
                for i in range(B):
                    cur_doc = ""
                    for j in doc_texts[i]:
                        cur_idx = j.item()
                        if cur_idx == self.ocr_dict['<END>']:
                            break
                        cur_doc += self.rev_ocr_dict[cur_idx]
                    docs.append(cur_doc)
            new_kie_logits = self.bilstm_layer(new_kie_logits, new_mask.sum(dim=-1), (None, None))
            if self.use_crf:
                if self.ins_lvl_mean:
                    log_likelihood = self.crf(new_kie_logits, new_tags, mask=new_mask)
                    if sorted_idx:
                        total_num_box = 0
                        for idx_set in sorted_idx:
                            total_num_box += len(idx_set)
                        log_likelihood /= total_num_box
                    else:
                        raise ValueError(f"sorted_idx is required for calculating the num of boxes")
                else:
                    log_likelihood = self.crf(new_kie_logits, new_tags, mask=new_mask)
                    log_likelihood /= new_mask.sum()
                if 'CRF' not in logits_logger:
                    logits_logger['CRF'] = [-log_likelihood.reshape(1)]
                else:
                    logits_logger['CRF'].append(-log_likelihood.reshape(1))
            else:
                if 'CRF' not in logits_logger:
                    logits_logger['CRF'] = [self.criterion(
                        new_kie_logits.reshape(-1, new_kie_logits.shape[-1]),
                        new_tags.reshape(-1))]
                else:
                    logits_logger['CRF'].append(self.criterion(
                        new_kie_logits.reshape(-1, new_kie_logits.shape[-1]),
                        new_tags.reshape(-1)))
        else:
            logits_logger['kie_logits'] = kie_logits
            if self.use_both_modal:
                logits_logger['rec_logits'] = rec_logits

    def debug_func(self, pred_texts, best_paths, sorted_idx):
        """
        Debug function.
        Args:
            pred_texts: BN, L
            best_paths: list[list[int]]
            sorted_idx: list[list[int]]
        Returns:

        """
        merged_texts = []
        for idx in sorted_idx[0]:
            for char_val in pred_texts[idx]:
                if char_val.item() == self.ocr_dict['<END>']:
                    break
                merged_texts.append(char_val.item())
        if len(merged_texts) != len(best_paths[0]):
            import ipdb
            ipdb.set_trace()
        # get IE results
        rev_entity_dict = dict()
        for key, val in self.entity_dict.items():
            rev_entity_dict[val] = key
        char_kv = dict()
        for entity_idx, char_idx in zip(best_paths[0], merged_texts):
            entity_tag = rev_entity_dict[entity_idx]
            if entity_tag != 'O':
                cls = entity_tag[2:]
                if cls not in char_kv:
                    char_kv[cls] = self.rev_ocr_dict[char_idx]
                else:
                    char_kv[cls] += self.rev_ocr_dict[char_idx]
        self.char_kv = char_kv
        for k, v in char_kv.items():
            print(f"{k} : {v}")
        import ipdb
        ipdb.set_trace()
        print(char_kv)

    def crf_decode(self, logits_logger, shape_, sorted_idx):
        if self.training:
            raise RuntimeError("crf_decode should be inference only")

        B, N = shape_
        C = logits_logger['kie_logits'].shape[-1]
        _, pred_texts = torch.max(self.soft_max_func(logits_logger['REC'][-1]), dim=-1)
        if self.use_both_modal:
            new_kie_logits, new_mask, mask = self.union_layer(
                torch.cat((logits_logger['kie_logits'], logits_logger['rec_logits']), dim=-1).reshape(B, N, -1, 2*C),
                pred_texts.reshape(B, N, -1),
                self.ocr_dict['<END>'],
                sorted_idx=sorted_idx)
        else:
            new_kie_logits, new_mask, mask = self.union_layer(
                logits_logger['kie_logits'].reshape(B, N, -1, C),
                pred_texts.reshape(B, N, -1),
                self.ocr_dict['<END>'],
                sorted_idx=sorted_idx)

        new_kie_logits = self.bilstm_layer(new_kie_logits, new_mask.sum(dim=-1), (None, None))
        if self.use_crf:
            # tmp = self.crf.viterbi_tags(new_kie_logits, new_mask, logits_batch_first=True)
            best_paths = self.crf.decode(new_kie_logits, new_mask)

            # self.debug_func(pred_texts, best_paths, sorted_idx)

            ins_lvl_tags = self.union_layer.split_out(best_paths, mask, sorted_idx=sorted_idx)
            if 'CRF' not in logits_logger:
                logits_logger['CRF'] = [ins_lvl_tags]
            else:
                logits_logger['CRF'].append(ins_lvl_tags)
        else:
            ins_lvl_logits = self.union_layer.split_logits_out(new_kie_logits, mask, sorted_idx=sorted_idx)
            if 'CRF' not in logits_logger:
                logits_logger['CRF'] = [ins_lvl_logits]
            else:
                logits_logger['CRF'].append(ins_lvl_logits)
