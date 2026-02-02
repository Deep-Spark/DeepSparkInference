set -euo pipefail

EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0));then
    echo "fails"
    EXIT_STATUS=1
    fi
}

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

echo "Fp16 Inference Acc!"
python3 inference_ixrt.py                        \
        --test_mode ACC                     \
        --model_path ${engine_file}   \
        --torken_path ./tokenizer_config/   \
        -i ./000000000139.jpg              \
        -t chair.person.tv ;  check_status
