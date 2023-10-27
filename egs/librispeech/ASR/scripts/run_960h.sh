# Librispeech 960h dataset training
# Offline baseline : Conformer(L)-stateless-T
./pruned_transducer_stateless5/train.py \
    --world-size 4 --num-epochs 30 --start-epoch 1 \
    --exp-dir pruned_transducer_stateless5/exp-960-baseline \
    --full-libri 1 --max-duration 350 --num-encoder-layers 17 \
    --dim-feedforward 2048 --nhead 8 --encoder-dim 512 \
    --decoder-dim 512 --joiner-dim 640 \

# 실험내용 : Dynamic chunk training
# training option중 --dynamic-chunk-training과 --causal-convolution을 1로 설정하여 DCT 학습을 진행
# --short-chunk-size는 일반적으로 0~25ms의 chunk size를 streaming 모델, 25ms~full context를 non streaming 모델이라고 함.
# 25ms일 때, context를 가장 많이 참조하기 때문에 보통 25ms의 DCT를 사용
# Streaming baseline : Conformer(L)-stateless-T
./pruned_transducer_stateless4/train.py \
    --world-size 4 --num-epochs 30 --start-epoch 1 \
    --exp-dir pruned_transducer_stateless5/exp-960-streaming-baseline \
    --full-libri 1 --max-duration 350 --num-encoder-layers 17 \
    --dim-feedforward 2048 --nhead 8 --encoder-dim 512 \
    --decoder-dim 512 --joiner-dim 640 \
    --dynamic-chunk-training 1 --causal-convolution 1 \
    --short-chunk-size 25 --num-left-chunks 4 \
    
# 실험내용 : Time-restricted based chunk-wise
# training 옵션의 num-left-chunk는 streaming decoding 시 left-context와 조율해야함 : left-context = decode-chunk-size * num-left-chunks
# acoustic feature의 stride가 30ms일 때, right-context를 한 frame만 적용한다면, num_layer * 30ms만큼의 latency가 발생
# Streaming baseline : Conformer(L)-stateless-T
./pruned_transducer_stateless5/train.py \
    --world-size 4 --num-epochs 30 --start-epoch 1 \
    --exp-dir pruned_transducer_stateless5/exp-960-streaming-left4 \
    --full-libri 1 --max-duration 350 --num-encoder-layers 17 \
    --dim-feedforward 2048 --nhead 8 --encoder-dim 512 \
    --decoder-dim 512 --joiner-dim 640 \
    --dynamic-chunk-training 1 --causal-convolution 1 \
    --short-chunk-size 25 --num-left-chunks 4 \

# 실험내용 : Delay penalization
# baseline 검증을 위해 icefall과 같은 모델 사이즈상에서 측정을 위해 ./pruned_transducer_stateless4의 conformer 모델 사용 
# Streaming baseline : Conformer(L)-stateless-T
./pruned_transducer_stateless4/train.py \
    --world-size 4 --num-epochs 30 --start-epoch 1 \
    --exp-dir pruned_transducer_stateless4/exp-960-streaming-p01 \
    --full-libri 1 --max-duration 350 --dynamic-chunk-training 1 \
    --causal-convolution 1 --short-chunk-size 25 --num-left-chunks 8 \
    --delay-penalty 0.01