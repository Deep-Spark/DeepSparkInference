pip install -r requirements.txt
mkdir -p ./python/data
ln -s /root/data/checkpoints/bert-large-uncased/ ./python/data && ln -s /root/data/datasets/squad/ ./python/data