export CUDA_VISIBLE_DEVICES="0,1,2,3"
for lang in "en" "es"; do
	./pruned_transducer_stateless_xlsr_v2/train.py \
		 --wandb False \
		 --input-strategy AudioSamples \
		 --enable-spec-aug False \
		 --multi-optim True \
		 --world-size 4 \
		 --num-epochs 30 \
		 --start-epoch 1 \
		 --exp-dir ./pruned_transducer_stateless_xlsr_v2/$lang \
		 --max-duration 25 \
		 --freeze-finetune-updates 800 \
		 --encoder-dim 1024 \
		 --decoder-dim 1024 \
		 --joiner-dim 1024 \
		 --use-fp16 1 \
		 --base-lr 0.01 \
		 --peak-dec-lr 0.04175 \
		 --peak-enc-lr 0.0003859 \
		 --accum-grads 16 \
		 --encoder-type xlsr \
		 --additional-block True \
		 --prune-range 5 \
		 --context-size 2 \
		 --ctc-loss-scale 0.2 \
		 --data-type commonvoice \
		 --lid False \
		 --lang-type $lang \
		 --decode-interval 100 \
		 --bpe-model data/$lang/lang_bpe_500/bpe.model
done
