# Librispeech 960h dataset training
# Offline baseline : Conformer(L)-stateless-T
./pruned_transducer_stateless5/train.py \
    --world-size 4 --num-epochs 30 --start-epoch 1 \
    --exp-dir pruned_transducer_stateless5/exp-960-baseline \
    --full-libri 1 --max-duration 350 --num-encoder-layers 17 \
    --dim-feedforward 2048 --nhead 8 --encoder-dim 512 \
    --decoder-dim 512 --joiner-dim 640 \

# 실험내용 : Dynamic chunk training
# Streaming baseline : Conformer(L)-stateless-T
./pruned_transducer_stateless5/train.py \
    --world-size 4 --num-epochs 30 --start-epoch 1 \
    --exp-dir pruned_transducer_stateless5/exp-960-streaming-baseline \
    --full-libri 1 --max-duration 350 --num-encoder-layers 17 \
    --dim-feedforward 2048 --nhead 8 --encoder-dim 512 \
    --decoder-dim 512 --joiner-dim 640 \
    --dynamic-chunk-training 1 --causal-convolution 1 \
    --short-chunk-size 25 --num-left-chunks 4 \
    
# 실험내용 : Time-retricted based chunk-wise
# issue : encoder layer의 수에 따라 left chunk가 볼 수 있는 범위 계산 공식 필요
# Streaming baseline : Conformer(L)-stateless-T
./pruned_transducer_stateless5/train.py \
    --world-size 4 --num-epochs 30 --start-epoch 1 \
    --exp-dir pruned_transducer_stateless5/exp-960-streaming-left8 \
    --full-libri 1 --max-duration 350 --num-encoder-layers 17 \
    --dim-feedforward 2048 --nhead 8 --encoder-dim 512 \
    --decoder-dim 512 --joiner-dim 640 \
    --dynamic-chunk-training 1 --causal-convolution 1 \
    --short-chunk-size 25 --num-left-chunks 8 \
    

# 실험내용 : Delay penalization
# issue : warmup 하이퍼파라미터에 따라 학습이 수렴이 안되는 문제 해결중
# Streaming baseline : Conformer(L)-stateless-T
./pruned_transducer_stateless5/train.py \
    --world-size 4 --num-epochs 30 --start-epoch 1 \
    --exp-dir pruned_transducer_stateless5/exp-960-streaming-delay01\
    --full-libri 1 --max-duration 350 --num-encoder-layers 17 \
    --dim-feedforward 2048 --nhead 8 --encoder-dim 512 \
    --decoder-dim 512 --joiner-dim 640 \
    --dynamic-chunk-training 1 --causal-convolution 1 \
    --short-chunk-size 25 --num-left-chunks 4 \
    --delay-penalty 0.1 \
