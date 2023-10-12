#!/usr/bin/env bash

# for method in greedy_search modified_beam_search fast_beam_search; do
# for method in fast_beam_search; do
#   ./pruned_transducer_stateless5/decode_grapheme.py \
#     --epoch 30 \
#     --avg 10 \
#     --exp-dir ./pruned_transducer_stateless5/exp-100 \
#     --max-duration 2000 \
#     --decoding-method $method \
#     --max-sym-per-frame 1 \
#     --num-encoder-layers 12 \
#     --dim-feedforward 1024 \
#     --nhead 8 \
#     --encoder-dim 512 \
#     --decoder-dim 512 \
#     --joiner-dim 640 \
#     --use-averaged-model True \
#     --bpe-model data/lang_bpe_550/bpe.model
# done

./pruned_transducer_stateless5/decode_grapheme.py \
    --epoch 30 \
    --avg 10 \
    --exp-dir ./pruned_transducer_stateless5/exp-100 \
    --max-duration 2000 \
    --num-encoder-layers 12 \
    --dim-feedforward 1024 \
    --nhead 8 \
    --encoder-dim 512 \
    --decoder-dim 512 \
    --joiner-dim 640 \
    --use-averaged-model True \
    --decoding-method fast_beam_search_nbest_LG \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --lang-dir data/lang_bpe_250 \
    --bpe-model data/lang_bpe_250/bpe.model