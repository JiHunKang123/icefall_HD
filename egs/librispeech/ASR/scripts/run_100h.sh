# Librispeech 100h dataset training
# Conformer(M)
./pruned_transducer_stateless5/train.py \
    --world-size 4 --num-epochs 30 --start-epoch 1 \
    --exp-dir pruned_transducer_stateless5/exp-100-M \
    --full-libri 0 --max-duration 350 --num-encoder-layers 12 \
    --dim-feedforward 2048 --nhead 8 --encoder-dim 512 \
    --decoder-dim 512 --joiner-dim 512 \

# Conformer (L)
./pruned_transducer_stateless5/train.py \
    --world-size 4 --num-epochs 30 --start-epoch 1 \
    --exp-dir pruned_transducer_stateless5/exp-100-L \
    --full-libri 0 --max-duration 350 --num-encoder-layers 17 \
    --dim-feedforward 2048 --nhead 8 --encoder-dim 512 \
    --decoder-dim 512 --joiner-dim 640

# 100시간으로 dynamic chunk trining시 학습이 제대로 되지 않음
# Streaming baseline : Conformer(L)-stateless-T
./pruned_transducer_stateless5/train.py \
    --world-size 4 --num-epochs 30 --start-epoch 1 \
    --exp-dir pruned_transducer_stateless5/exp-100-streaming-baseline \
    --full-libri 0 --max-duration 350 --num-encoder-layers 17 \
    --dim-feedforward 2048 --nhead 8 --encoder-dim 512 \
    --decoder-dim 512 --joiner-dim 640 \
    --dynamic-chunk-training 1 --causal-convolution 1 \
    --short-chunk-size 25 --num-left-chunks 4 \
