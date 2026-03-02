set -euo pipefail

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
    echo "fails"
    EXIT_STATUS=1
    fi
}

fps_target=-1

# Update arguments
index=0
options=$@
arguments=($options)
for argument in $options
do
    index=`expr $index + 1`
    case $argument in
      --tgt) fps_target=${arguments[index]};;
    esac
done

onnx_model=./checkpoints/grounded_static_1x800x1200x128_sim_end.onnx
engine_file=./checkpoints/grounded_static_1x800x1200x128_sim_end.engine

if [ -f $engine_file ];then
    echo "  "Build Engine Skip, $engine_file has been existed
else
    echo "Build Fp16 Engine!"
    python3 build_engine.py             \
            --precision float16         \
            --model ${onnx_model}       \
            --engine ${engine_file}; check_status
fi

echo "Fp16 Inference Fps!"
python3 inference_ixrt.py                        \
        --test_mode FPS                     \
        --model_path ${engine_file}   \
        --fps_target ${fps_target}    \
        --torken_path ./tokenizer_config/   \
        -i ./000000000139.jpg              \
        -t chair.person.tv ;  check_status
