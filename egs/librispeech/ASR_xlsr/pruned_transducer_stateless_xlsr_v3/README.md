# Spoken Language Identification

- [English](https://commonvoice.mozilla.org/en/datasets), [Spanish](https://commonvoice.mozilla.org/es/datasets): CommonVoice 15.0
- Korean:  HD100
- dev accuracy(%)

| en | es | ko | avg |
| --- | --- | --- | --- |
| 89.26 | 95.89 | 99.55 | 94.90 |

- test accuracy(%)

| en | es | ko | avg |
| --- | --- | --- | --- |
| 85.50 | 93.57 | 99.58 | 92.89 |


# Installation

- Dataset preparation

1. CommonVoice dataset prepare(for English, Spanish)

```bash
cd icefall_HD/egs/commonvoice/ASR
./prepare_en.sh
./prepare_es.sh

# move data folder to icefall_HD/egs/librispeech/ASR/data/{language} or symbolic link ..
```

2. HD100 dataset prepare(for Korean)

```bash
cd icefall_HD/egs/hd100/ASR_xlsr
./0_prepare_hd100_grapheme.sh

# for name edit.. -> This can be skipped #
cd data/fbank
mv librispeech_cuts_train-clean-100.jsonl.gz hd_100_cuts_train-clean-100.jsonl.gz
mv librispeech_cuts_dev-clean.jsonl.gz hd_100_cuts_dev-clean.jsonl.gz
mv librispeech_cuts_test-clean.jsonl.gz hd_100_cuts_test-clean.jsonl.gz
##### This can be skipped #####

# move data folder to icefall_HD/egs/librispeech/ASR/data/ko or symbolic link ..
```

3. Full dataset prepare(for train lid)
	- This step **must be performed** after performing steps 1 and 2 above.

```bash
cd icefall_HD/egs/librispeech/ASR/data
mkdir full/fbank

cd ../en/fbank
cp cv-en_cuts_*.jsonl.gz ../../full/fbank

cd ../es/fbank
cp cv-es_cuts_*.jsnol.gz ../../full/fbank

cd ../ko/fbank
cp hd100_cuts_*.jsonl.gz ../../full/fbank

# create new train, dev, and test jsonl.gz files
cat <en-train.jsonl.gz> <es-train.jsonl.gz> <ko-train.jsonl.gz> > cv-full_cuts_train.jsonl.gz
cat <en-dev.jsonl.gz> <es-dev.jsonl.gz> <ko-dev.jsonl.gz> > cv-full_cuts_dev.jsonl.gz
cat <en-test.jsonl.gz> <es-test.jsonl.gz> <ko-test.jsonl.gz> > cv-full_cuts_test.jsonl.gz

# shuffle data within each file -> preparation done!
```

- LID train command

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless_xlsr_v2/train.py \
 --wandb False \
 --input-strategy AudioSamples \
 --enable-spec-aug False \
 --multi-optim False \
 --world-size 4 \
 --num-epochs 30 \
 --start-epoch 1 \
 --exp-dir ./pruned_transducer_stateless_xlsr_v2/tmp \
 --max-duration 25 \
 --freeze-finetune-updates 0 \
 --encoder-dim 1024 \
 --decoder-dim 1024 \
 --joiner-dim 1024 \
 --use-fp16 1 \
 --base-lr 0.01 \
 --peak-dec-lr 0.04175 \
 --peak-enc-lr 0.0003859 \
 --accum-grads 8 \
 --encoder-type xlsr \
 --additional-block True \
 --prune-range 5 \
 --context-size 2 \
 --ctc-loss-scale 0 \
 --data-type commonvoice \
 --lid True \
 --decode-interval 30000 \
 --bpe-model data/en/lang_bpe_500/bpe.model
```

- LID decode command

```bash
for method in lid; do
	./pruned_transducer_stateless_xlsr_v2/decode.py \
	  --input-strategy AudioSamples \
	  --enable-spec-aug False \
	  --additional-block True \
	  --model-name best-train-loss.pt \
	  --exp-dir ./pruned_transducer_stateless_xlsr_v2/tmp \
	  --max-duration 25 \
	  --decoding-method $method \
	  --max-sym-per-frame 1 \
	  --encoder-type xlsr \
	  --encoder-dim 1024 \
	  --decoder-dim 1024 \
	  --joiner-dim 1024 \
	  --decode-data-type commonvoice \
	  --lid True \
	  --bpe-model data/en/lang_bpe_500/bpe.model
done
```

- After lid model, en-xlsr, es-xlsr, ko-xlsr is ready: decoding command
	- Models should be ready in **{exp-dir}/{model-name}** 
```bash
for method in greedy_search; do
	./pruned_transducer_stateless_xlsr_v3/decode.py \
	  --input-strategy AudioSamples \
	  --enable-spec-aug False \
	  --additional-block True \
	  --exp-dir /DB/results/icefall \
	  --max-duration 25 \
	  --decoding-method $method \
	  --max-sym-per-frame 1 \
	  --encoder-type xlsr \
	  --encoder-dim 1024 \
	  --decoder-dim 1024 \
	  --joiner-dim 1024 \
	  --decode-data-type commonvoice \
	  --lid True \
	  --language-num 3 \
	  --bucketing-sampler False \
	  --model-name xlsr_lid_ko_hd100_allsec/best-train-loss.pt,en/epoch-30.pt,es/epoch-30.pt,ko/epoch-30.pt \
	  --bpe-model data/en/lang_bpe_500/bpe.model,data/es/lang_bpe_500/bpe.model,data/ko/lang_bpe_250/bpe.model \
done
```

