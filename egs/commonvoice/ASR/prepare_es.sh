#!/usr/bin/env bash

set -eou pipefail

nj=16
stage=$1
#stage=-1
stop_stage=100

# Split data/${lang}set to this number of pieces
# This is to avoid OOM during feature extraction.
#num_splits=2
num_splits=1000

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/$release/$lang
#      This directory contains the following files downloaded from
#       https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/${release}/${release}-${lang}.tar.gz
#
#     - clips
#     - dev.tsv
#     - invalidated.tsv
#     - other.tsv
#     - reported.tsv
#     - test.tsv
#     - train.tsv
#     - validated.tsv
#
#  - $dl_dir/musan
#      This directory contains the following directories downloaded from
#       http://www.openslr.org/17/
#
#     - music
#     - noise
#     - speech

#dl_dir=$PWD/download
dl_dir=/DB/CommonVoice
release=cv-corpus-15.0-2023-09-08
#release=cv-corpus-13.0-2023-03-09
lang=es	#ko, es

. shared/parse_options.sh || exit 1

# vocab size for sentence piece models.
# It will generate data/${lang}/lang_bpe_xxx,
# data/${lang}/lang_bpe_yyy if the array contains xxx, yyy
vocab_sizes=(
  # 5000
  # 2000
  # 1000
  500
  #50
)

# All files generated by this script are saved in "data/${lang}".
# You can safely remove "data/${lang}" and rerun this script to regenerate it.
mkdir -p data/${lang}

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if ! command -v ffmpeg &> /dev/null; then
  echo "This dataset requires ffmpeg"
  echo "Please install ffmpeg first"
  echo ""
  echo "  sudo apt-get install ffmpeg"
  exit 1
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  # If you have pre-downloaded it to /path/to/$release,
  # you can create a symlink
  #
  #   ln -sfv /path/to/$release $dl_dir/$release
  #
  if [ ! -d $dl_dir/$release/$lang/clips ]; then
    lhotse download commonvoice --languages $lang --release $release $dl_dir
  fi

  # If you have pre-downloaded it to /path/to/musan,
  # you can create a symlink
  #
  #   ln -sfv /path/to/musan $dl_dir/
  #
  if [ ! -d $dl_dir/musan ]; then
    lhotse download musan $dl_dir
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare CommonVoice manifest"
  # We assume that you have downloaded the CommonVoice corpus
  # to $dl_dir/$release
  mkdir -p data/${lang}/manifests
  if [ ! -e data/${lang}/manifests/.cv-${lang}.done ]; then
    lhotse prepare commonvoice --language $lang -j $nj $dl_dir/$release data/${lang}/manifests
    touch data/${lang}/manifests/.cv-${lang}.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to data/musan
  mkdir -p data/manifests
  if [ ! -e data/manifests/.musan.done ]; then
    lhotse prepare musan $dl_dir/musan data/manifests
    touch data/manifests/.musan.done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Preprocess CommonVoice manifest"
  if [ ! -e data/${lang}/fbank/.preprocess_complete ]; then
    ./local/preprocess_commonvoice.py  --language $lang
    touch data/${lang}/fbank/.preprocess_complete
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank for dev and test subsets of CommonVoice"
  mkdir -p data/${lang}/fbank
  if [ ! -e data/${lang}/fbank/.cv-${lang}_dev_test.done ]; then
    ./local/compute_fbank_commonvoice_dev_test.py --language $lang
    touch data/${lang}/fbank/.cv-${lang}_dev_test.done
  fi
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Split train subset into ${num_splits} pieces"
  split_dir=data/${lang}/fbank/cv-${lang}_train_split_${num_splits}
  if [ ! -e $split_dir/.cv-${lang}_train_split.done ]; then
    lhotse split $num_splits ./data/${lang}/fbank/cv-${lang}_cuts_train_raw.jsonl.gz $split_dir
    touch $split_dir/.cv-${lang}_train_split.done
  fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Compute features for train subset of CommonVoice"
  if [ ! -e data/${lang}/fbank/.cv-${lang}_train.done ]; then
    ./local/compute_fbank_commonvoice_splits.py \
      --num-workers $nj \
      --batch-duration 600 \
      --start 0 \
      --num-splits $num_splits \
      --language $lang
    touch data/${lang}/fbank/.cv-${lang}_train.done
  fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Combine features for train"
  if [ ! -f data/${lang}/fbank/cv-${lang}_cuts_train.jsonl.gz ]; then
    pieces=$(find data/${lang}/fbank/cv-${lang}_train_split_${num_splits} -name "cv-${lang}_cuts_train.*.jsonl.gz")
    lhotse combine $pieces data/${lang}/fbank/cv-${lang}_cuts_train.jsonl.gz
  fi
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 8: Compute fbank for musan"
  mkdir -p data/fbank
  if [ ! -e data/fbank/.musan.done ]; then
    ./local/compute_fbank_musan.py
    touch data/fbank/.musan.done
  fi
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Prepare BPE based lang"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/${lang}/lang_bpe_${vocab_size}
    mkdir -p $lang_dir

    if [ ! -f $lang_dir/transcript_words.txt ]; then
      log "Generate data for BPE training"
      file=$(
        find "data/${lang}/fbank/cv-${lang}_cuts_train.jsonl.gz"
      )
      gunzip -c ${file} | awk -F '"' '{print $30}' > $lang_dir/transcript_words.txt

      # Ensure space only appears once
      sed -i 's/\t/ /g' $lang_dir/transcript_words.txt
      sed -i 's/[ ][ ]*/ /g' $lang_dir/transcript_words.txt
    fi
 
    if [ ! -f $lang_dir/words.txt ]; then
      cat $lang_dir/transcript_words.txt | sed 's/ /\n/g' \
        | sort -u | sed '/^$/d' > $lang_dir/words.txt
      (echo '!SIL'; echo '<SPOKEN_NOISE>'; echo '<UNK>'; ) |
        cat - $lang_dir/words.txt | sort | uniq | awk '
        BEGIN {
          print "<eps> 0";
        }
        {
          if ($1 == "<s>") {
            print "<s> is in the vocabulary!" | "cat 1>&2"
            exit 1;
          }
          if ($1 == "</s>") {
            print "</s> is in the vocabulary!" | "cat 1>&2"
            exit 1;
          }
          printf("%s %d\n", $1, NR);
        }
        END {
          printf("#0 %d\n", NR+1);
          printf("<s> %d\n", NR+2);
          printf("</s> %d\n", NR+3);
        }' > $lang_dir/words || exit 1;
      mv $lang_dir/words $lang_dir/words.txt
    fi
 
    if [ ! -f $lang_dir/bpe.model ]; then
      ./local/train_bpe_model.py \
        --lang-dir $lang_dir \
        --vocab-size $vocab_size \
        --transcript $lang_dir/transcript_words.txt
    fi
  
    if [ ! -f $lang_dir/L_disambig.pt ]; then
      ./local/prepare_lang_bpe.py --lang-dir $lang_dir

      log "Validating $lang_dir/lexicon.txt"
      ./local/validate_bpe_lexicon.py \
        --lexicon $lang_dir/lexicon.txt \
        --bpe-model $lang_dir/bpe.model
    fi

    if [ ! -f $lang_dir/L.fst ]; then
      log "Converting L.pt to L.fst"
      ./shared/convert-k2-to-openfst.py \
        --olabels aux_labels \
        $lang_dir/L.pt \
        $lang_dir/L.fst
    fi

    if [ ! -f $lang_dir/L_disambig.fst ]; then
      log "Converting L_disambig.pt to L_disambig.fst"
      ./shared/convert-k2-to-openfst.py \
        --olabels aux_labels \
        $lang_dir/L_disambig.pt \
        $lang_dir/L_disambig.fst
    fi
  done
fi
