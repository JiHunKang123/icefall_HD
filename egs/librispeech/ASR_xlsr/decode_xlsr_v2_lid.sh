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