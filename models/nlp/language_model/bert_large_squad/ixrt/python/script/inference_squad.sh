BSZ=1
USE_FP16=True

# Update arguments
index=0
options=$@
arguments=($options)
for argument in $options
do
    index=`expr $index + 1`
    case $argument in
      --bs) BSZ=${arguments[index]};;
      --int8) USE_FP16=False;;
    esac
done

if [ "$USE_FP16" = "True" ]; then
    echo 'USE_FP16=True'
    UMD_ENABLEDCPRINGNUM=16 python3 inference.py -e ./data/bert_large_384.engine \
                            -b ${BSZ} \
                            -s 384 \
                            -sq ./data/squad/dev-v1.1.json \
                            -v ./data/bert-large-uncased/vocab.txt \
                            -o ./data/predictions-bert_large_384.json 
    python3 evaluate-v1.1.py  ./data/squad/dev-v1.1.json  ./data/predictions-bert_large_384.json 90
else
    echo 'USE_INT8=True'
    UMD_ENABLEDCPRINGNUM=16 python3 inference.py -e ./data/bert_large_384_int8.engine \
                            -b ${BSZ} \
                            -s 384 \
                            -sq ./data/squad/dev-v1.1.json \
                            -v ./data/bert-large-uncased/vocab.txt \
                            -o ./data/predictions-bert_large_384_int8.json \
                            -i
    python3 evaluate-v1.1.py  ./data/squad/dev-v1.1.json  ./data/predictions-bert_large_384_int8.json 88
fi