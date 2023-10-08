# offline models decoding script
# greedy_search
./pruned_transducer_stateless5/decode.py \
--epoch 25 --avg 10 \
--exp-dir ./pruned_transducer_stateless5/exp-100-L \
--max-duration 600 --decoding-method greedy_search \
--num-encoder-layers 17 --dim-feedforward 2048 \
--nhead 8 --encoder-dim 512 --decoder-dim 512 --joiner-dim 640 \


# streaming models decoding script
./pruned_transducer_stateless5/decode.py \
    --epoch 25 --avg 10 \
    --exp-dir ./pruned_transducer_stateless5/exp-960-streaming \
    --max-duration 600 --decoding-method greedy_search \
    --num-encoder-layers 17 --dim-feedforward 2048 \
    --nhead 8 --encoder-dim 512 --decoder-dim 512 --joiner-dim 640 \
    --decode-chunk-len 32 \