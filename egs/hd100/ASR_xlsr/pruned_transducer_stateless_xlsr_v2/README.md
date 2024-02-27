# xlsr-transducer

|  | test-clean | test-other |
| --- | --- | --- |
| greedy decoding | 4.90 | 11.31 |
| modified beam search | 4.54 | 10.52 |
| fast beam search | 4.7 | 10.84 |
- train command

```bash
./pruned_transducer_stateless_xlsr_v2/train.py
	--wandb False \
	--input-strategy AudioSamples \
	--enable-spec-aug False \
	--multi-optim True \
	--start-epoch 1 \
	--world-size 4 \
	--num-epochs 30 \
	--full-libri 0 \
	--exp-dir ./pruned_transducer_stateless_xlsr_v2/tmp \
	--max-duration 50 \
	--freeze-finetune-updates 800 \
	--encoder-dim 1024 \
	--decoder-dim 1024 \
	--joiner-dim 1024 \
	--use-fp16 1 \
	--peak-dec-lr 0.04175 \
	--peak-enc-lr 0.0003859 \
	--accum-grads 8 \
	--encoder-type xlsr \
	--additional-block True \
	--prune-range 5 \
	--context-size 2 \
	--ctc-loss-scale 0.2
```

- decode command

```bash
for method in greedy_search modified_beam_search fast_beam_search; do
  ./pruned_transducer_stateless_xlsr_v2/decode.py \
    --input-strategy AudioSamples \
    --enable-spec-aug False \
    --additional-block True \
    --model-name epoch-30.pt \
    --exp-dir ./pruned_transducer_stateless_xlsr_v2/tmp \
    --max-duration 400 \
    --decoding-method $method \
    --max-sym-per-frame 1 \ 
    --encoder-type xlsr \
    --encoder-dim 1024 \
    --decoder-dim 1024 \
    --joiner-dim 1024
done
```
