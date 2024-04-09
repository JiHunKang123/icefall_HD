for lang in "en"; do
	for method in greedy_search modified_beam_search fast_beam_search; do
		./pruned_transducer_stateless_xlsr_v2/decode.py \
			--input-strategy AudioSamples \
		    --enable-spec-aug False \
		    --additional-block True \
		    --model-name epoch-30.pt \
		    --exp-dir /DB/results/icefall/$lang \
		    --max-duration 25 \
		    --decoding-method $method \
		    --max-sym-per-frame 1 \
		    --encoder-type xlsr \
		    --encoder-dim 1024 \
		    --decoder-dim 1024 \
		    --joiner-dim 1024 \
		    --decode-data-type commonvoice \
		    --lid False \
		    --lang-type $lang \
		    --bpe-model data/$lang/lang_bpe_500/bpe.model
	done
done
