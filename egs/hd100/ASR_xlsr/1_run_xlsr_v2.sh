export CUDA_VISIBLE_DEVICES="0,1,2,3"
vocab_sizes=250

./pruned_transducer_stateless_xlsr_v2/train.py \
 --wandb False \
 --input-strategy AudioSamples \
 --enable-spec-aug False \
 --multi-optim True \
 --world-size 4 \
 --num-epochs 30 \
 --start-epoch 1 \
 --exp-dir ./pruned_transducer_stateless_xlsr_v2/tmp \
 --max-duration 80 \
 --freeze-finetune-updates 800 \
 --encoder-dim 1024 \
 --decoder-dim 1024 \
 --joiner-dim 1024 \
 --use-fp16 1 \
 --peak-dec-lr 0.04175 \
 --peak-enc-lr 0.0003859 \
 --accum-grads 6 \
 --encoder-type xlsr \
 --additional-block True \
 --prune-range 5 \
 --context-size 2 \
 --ctc-loss-scale 0.2 \
 --bpe-model data/lang_bpe_${vocab_sizes}/bpe.model
