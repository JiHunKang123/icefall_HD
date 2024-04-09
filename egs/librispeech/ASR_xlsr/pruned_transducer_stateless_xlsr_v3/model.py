# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from typing import Tuple
import logging

import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from scaling import ScaledLinear

from icefall.utils import add_sos


class Transducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder: EncoderInterface,
        decoder: nn.Module,
        joiner: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
        lid=False,
        language_num=1,
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dm) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder, "blank_id") or hasattr(decoder[0], "blank_id")
        self.language_num = language_num

        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner

        #self.simple_am_proj = nn.Linear(
        #    encoder_dim,
        #    vocab_size,
        #)
        
        if language_num == 1:
            self.simple_am_proj = ScaledLinear(
                encoder_dim,
                vocab_size,
            )
            self.simple_lm_proj = ScaledLinear(decoder_dim, vocab_size)
        
            self.ctc_output = nn.Sequential(
                nn.Dropout(p=0.1),
                ScaledLinear(encoder_dim, vocab_size),
                #nn.Linear(encoder_dim, vocab_size),
                nn.LogSoftmax(dim=-1),
            )
        else:
            self.ctc_output = [nn.Sequential(
                nn.Dropout(p=0.0),
                ScaledLinear(encoder_dim, vocab_size[i]),
                #nn.Linear(encoder_dim, vocab_size),
                nn.LogSoftmax(dim=-1),
            ) for i in range(language_num)]

        if lid == True:
            self.lid = True
            self.lstm = nn.LSTM(512, 256, 1, batch_first=True)
            self.lid_linear = nn.Linear(256, 3)
            self.softmax = nn.Softmax(dim=1)
            #self.ce_loss = nn.CrossEntropyLoss(reduction='none')
            self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

        else:
            self.lid = False

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        target_lang: torch.Tensor = None,
        #target_en,
        #target_es,
        #target_ko,
   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return a tuple containing simple loss, pruned loss, and ctc-output.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 2 or x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

        # arg 추가해서 짜주기 #
        if self.language_num == 1:
            encoder_out, x_lens, cnn_out = self.encoder(x, x_lens)
        else:
            encoder_out, x_lens, cnn_out = self.encoder(x, x_lens, self.lstm, self.lid_linear, self.softmax)
        #encoder_out, x_lens, cnn_out = self.encoder(x, x_lens)
        #encoder_out, x_lens = self.encoder(x, x_lens)
        assert torch.all(x_lens > 0)

        #For LID
        if self.lid == True:
            cnn_out = cnn_out.transpose(1, 2)
            output = self.lstm(cnn_out)
            #print(output[0])
            
            # x_lens == olens
            final = []
            final = torch.tensor(final).to('cuda')
            
            # for 2 seconds lid
            #final = output[0][:, 100, :]

            # for all seconds lid
            for i in range(len(x_lens)):
                new_output = output[0][i, x_lens[i]-1, :]
                new_output = new_output.reshape(1, -1)
                final = torch.cat((final, new_output), dim=0)

            lid_final = self.lid_linear(final)
            lid_final = self.softmax(lid_final)
            #lid_prob = self.softmax(lid_final)
            
            ### Compute CE Loss
            ce_loss = self.ce_loss(lid_final, target_lang)
            #prob = lid_final.max(dim=1)[0]
            #pred = lid_final.max(dim=1)[1]

            ### Compute CE Loss per language
            """
            new_tar_en, new_tar_es, new_tar_ko = 0, 0, 0
            
            if target_ko == 0:
                multiples = target_en * target_es
            if target_en == 0:
                multiples = target_es * target_ko
            if target_es == 0:
                multiples = target_en * target_ko
            if target_ko == 0 and target_es == 0:
                multiples = target_en
            if target_en == 0 and target_es == 0:
                multiples = target_ko
            if target_ko == 0 and target_en == 0:
                multiples = target_es
            if target_ko != 0 and target_en != 0 and target_es != 0:
                multiples = target_en * target_es * target_ko
           
            try: new_tar_en = multiples / target_en
            except: new_tar_en = 0
            try: new_tar_es = multiples / target_es
            except: new_tar_es = 0
            try: new_tar_ko = multiples / target_ko
            except: new_tar_ko = 0
            
            denom = new_tar_en + new_tar_es + new_tar_ko
           
            ratio_en = new_tar_en / denom
            ratio_es = new_tar_es / denom
            ratio_ko = new_tar_ko / denom
            
            for i, t_ln in enumerate(target_lang):
                if t_ln == 0:
                    ce_loss[i] = ce_loss[i] * ratio_en
                elif t_ln == 1:
                    ce_loss[i] = ce_loss[i] * ratio_es
                elif t_ln == 2:
                    ce_loss[i] = ce_loss[i] * ratio_ko

            ce_loss = sum(ce_loss) / len(ce_loss)
            """
            num_corrects = (torch.max(lid_final, 1)[1].view(target_lang.size()).data == target_lang.data).float().sum()
            acc = 100 * num_corrects / lid_final.size(0)
            
            if random.random() < 0.1:
                logging.info(f'acc: {acc}')
                '''
                logging.info(prob)
                logging.info(pred)
                logging.info(target_lang)
                logging.info(ce_loss)
                '''
        else:
            ce_loss = None

        # compute ctc log-probs
        ctc_output = self.ctc_output(encoder_out)

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros((x.size(0), 4), dtype=torch.int64, device=x.device)
        boundary[:, 2] = y_lens
        boundary[:, 3] = x_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return (simple_loss, pruned_loss, ctc_output, ce_loss)

    def decode(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        sp,
    ):
        from beam_search import greedy_search_batch

        encoder_out, x_lens, cnn_out = self.encoder(x, x_lens)
        #encoder_out, x_lens = self.encoder(x, x_lens)

        hyps = []
        hyp_tokens = greedy_search_batch(self, encoder_out, x_lens)

        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())

        return hyps

