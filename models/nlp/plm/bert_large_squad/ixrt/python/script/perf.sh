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
    python3 perf.py -e ./data/bert_large_384.engine -b ${BSZ} -s 384
else
    echo 'USE_INT8=True'
    python3 perf.py -e ./data/bert_large_384_int8.engine -b ${BSZ} -s 384
fi