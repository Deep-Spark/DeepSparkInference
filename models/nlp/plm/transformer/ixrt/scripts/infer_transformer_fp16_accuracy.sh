set -euo pipefail

EXIT_STATUS=0
check_status()
{
    ret_code=${PIPESTATUS[0]}
    if [ ${ret_code} != 0 ]; then
    echo "fails"
    [[ ${ret_code} -eq 10 && "${TEST_PERF:-1}" -eq 0 ]] || EXIT_STATUS=1
    fi
}

BATCH_SIZE=${BATCH_SIZE:=128}
# Update arguments
index=0
options=$@
arguments=($options)
for argument in $options
do
    index=`expr $index + 1`
    case $argument in
      --bs) BATCH_SIZE=${arguments[index]};;
      --tgt) Accuracy=${arguments[index]};;
    esac
done

current_path=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
DATA_DIR=${current_path}/../data/datasets/wmt14.en-fr.joined-dict.newstest2014
MODEL_DIR=${current_path}/../data/checkpoints/wmt14.en-fr.joined-dict.transformer
CPU_AFFINITY=$(ixsmi topo -m|grep "^GPU0" |awk '{print $(NF-1)}')

if [[ ! -f "${MODEL_DIR}/Encoder.engine" ||  ! -f "${MODEL_DIR}/Decoder.engine" ]]; then
    echo "Build Engine."
    python3 ../plugin/build_engine.py \
        --model_dir ${MODEL_DIR}  
fi

echo "Inference(Test Accuracy)"
export Accuracy=${Accuracy:=42}
numactl --physcpubind=${CPU_AFFINITY} python3 inference_wmt14_en_fr_fp16_accuracy_plugin.py  ${DATA_DIR}  \
    --path ${MODEL_DIR}/model.pt \
    --beam 1 --batch-size ${BATCH_SIZE} \
    --remove-bpe --quiet --fp16; check_status;
exit ${EXIT_STATUS}
