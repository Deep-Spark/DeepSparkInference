set -eo pipefail

BSZ=32
TGT=200
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
datasets_path=${project_path}/data/datasets

echo 'USE_TRT='${USE_TRT}
export USE_TRT=$USE_TRT

echo "Step1 Build Engine Int8(bert large squad)!"
python3 builder_int8.py -pt ${checkpoints_path}/bert_large_int8_qat.bin \
                -o ${checkpoints_path}/bert_large_int8_b${BSZ}.engine \
                -b 1 ${BSZ} ${BSZ} \
                -s 1 384 384 \
                -i \
                -c ${checkpoints_path}

echo "Step2 Inference(test QPS)"
UMD_ENABLEDCPRINGNUM=16 python3 inference.py -e ${checkpoints_path}/bert_large_int8_b${BSZ}.engine \
                        -b ${BSZ} \
                        -s 384 \
                        -sq ${datasets_path}/squad/dev-v1.1.json \
                        -v ${checkpoints_path}/vocab.txt \
                        -o ${checkpoints_path}/predictions-bert_large_int8_b${BSZ}.json \
                        -z ${USE_TRT} \
                        --target_qps ${TGT} \
                        -i         