pip install -r requirements.txt
mkdir -p ./python/data
ln -s /root/data/checkpoints/bert_base_uncased_squad/ ./python/data && ln -s /root/data/datasets/squad/ ./python/data