PASSAGE='TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, 
speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations 
for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components 
to take advantage of powerful TensorRT optimizations for your apps.'
QUESTION="What is TensorRT?"

USE_FP16=True

# Update arguments
index=0
options=$@
arguments=($options)
for argument in $options
do
    index=`expr $index + 1`
    case $argument in
      --int8) USE_FP16=False;;
    esac
done

if [ "$USE_FP16" = "True" ]; then
    echo 'USE_FP16=True'
    python3 inference.py -e ./data/bert_large_384.engine \
                        -s 384 \
                        -p $PASSAGE \
                        -q $QUESTION \
                        -v ./data/bert-large-uncased/vocab.txt 
else
    echo 'USE_INT8=True'
    python3 inference.py -e ./data/bert_large_384_int8.engine \
                        -s 384 \
                        -p $PASSAGE \
                        -q $QUESTION \
                        -v ./data/bert-large-uncased/vocab.txt 
fi

