set -eo pipefail

BSZ=32
TGT=90
USE_TRT=False

# Update arguments
index=0
options=$@
arguments=($options)
for argument in $options
do
    index=`expr $index + 1`
    case $argument in
      --bs) BSZ=${arguments[index]};;
      --tgt) TGT=${arguments[index]};;
      --use_trt) USE_TRT=${arguments[index]};;
    esac
done
project_path=./
checkpoints_path=${project_path}/data/checkpoints/bert-large-uncased
datasets_path=${project_path}/data/datasets/squad

echo 'USE_TRT='${USE_TRT}
export USE_TRT=$USE_TRT

echo "Step1 Build Engine FP16(bert large squad)!"
python3 builder.py -x ${checkpoints_path}/bert_large_v1_1_fake_quant.onnx \
                   -w 4096 \
                   -o ${checkpoints_path}/bert_large_b${BSZ}.engine \
                   -s 1 384 384\
                   -b 1 ${BSZ} ${BSZ}\
                   --fp16 \
                   -c ${checkpoints_path}/bert_config.json \
                   -z ${USE_TRT}

echo "Step2 Run dev.json and generate json"
python3 inference.py -e ${checkpoints_path}/bert_large_b${BSZ}.engine \
                        -s 384 \
                        -b ${BSZ} \
                        -sq ${datasets_path}/squad/dev-v1.1.json \
                        -v ${checkpoints_path}/vocab.txt \
                        -o ${checkpoints_path}/predictions-bert_large_b${BSZ}.json \
                        -z ${USE_TRT}

echo "Step3 Inference(test F1-score)"
python3 evaluate-v1.1.py  ${datasets_path}/squad/dev-v1.1.json  ${checkpoints_path}/predictions-bert_large_b${BSZ}.json ${TGT}