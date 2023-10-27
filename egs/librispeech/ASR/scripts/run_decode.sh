# offline models decoding script
# greedy_search
./pruned_transducer_stateless5/decode.py \
--epoch 25 --avg 10 \
--exp-dir ./pruned_transducer_stateless5/exp-100-L \
--max-duration 600 --decoding-method greedy_search \
--num-encoder-layers 17 --dim-feedforward 2048 \
--nhead 8 --encoder-dim 512 --decoder-dim 512 --joiner-dim 640 \


# streaming models decoding script
# --decode-chunk-size : decoding 때 frame의 chunk size ex) 32 -> 320ms
./pruned_transducer_stateless5/decode.py \
    --epoch 25 --avg 10 \
    --exp-dir ./pruned_transducer_stateless5/exp-960-streaming \
    --max-duration 600 --decoding-method greedy_search \
    --num-encoder-layers 17 --dim-feedforward 2048 \
    --nhead 8 --encoder-dim 512 --decoder-dim 512 --joiner-dim 640 \
    --decode-chunk-len 32 \

# greedy_search
./pruned_transducer_stateless5/streaming_decode.py \
    --epoch 30 --avg 10 \
    --left-context 64 --decode-chunk-size 16 --right-context 0 \
    --exp-dir ./pruned_transducer_stateless5/exp-960-streaming-left4 \
    --num-encoder-layers 17 --dim-feedforward 2048 \
    --nhead 8 --encoder-dim 512 --decoder-dim 512 --joiner-dim 640 \
    --decoding-method greedy_search --num-decode-streams 200

# fast_beam_search
./pruned_transducer_stateless5/streaming_decode.py \
    --epoch 30 --avg 10 \
    --left-context 64 --decode-chunk-size 16 --right-context 0 \
    --exp-dir ./pruned_transducer_stateless5/exp-960-streaming-left4 \
    --num-encoder-layers 17 --dim-feedforward 2048 \
    --nhead 8 --encoder-dim 512 --decoder-dim 512 --joiner-dim 640 \
    --decoding-method fast_beam_search --num-decode-streams 200

# modified_beam_search
./pruned_transducer_stateless5/streaming_decode.py \
    --epoch 30 --avg 10 \
    --left-context 64 --decode-chunk-size 16 --right-context 0 \
    --exp-dir ./pruned_transducer_stateless5/exp-960-streaming-left4 \
    --num-encoder-layers 17 --dim-feedforward 2048 \
    --nhead 8 --encoder-dim 512 --decoder-dim 512 --joiner-dim 640 \
    --decoding-method modified_beam_search --num-decode-streams 200

./pruned_transducer_stateless4/streaming_decode.py \
    --epoch 30 --avg 10 \
    --left-context 64 --decode-chunk-size 16 --right-context 0 \
    --exp-dir ./pruned_transducer_stateless4/exp-960-streaming-p01 \
    --decoding-method greedy_search --num-decode-streams 200