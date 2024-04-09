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
	  #--exp-dir ./pruned_transducer_stateless_xlsr_v2/xlsr_lid_eql_2sec_60epoch \
done

#--model-name best-train-loss.pt \
#--exp-dir /workspace/icefall_hd/egs/librispeech/ASR/pruned_transducer_stateless_xlsr_v2/xlsr_lid_ko_hd100_allsec \
