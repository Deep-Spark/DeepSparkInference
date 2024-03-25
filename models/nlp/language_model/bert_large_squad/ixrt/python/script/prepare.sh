VERSION='v1.1'

while test $# -gt 0
do
    case "$1" in
        -h) echo "Usage: sh download_squad.sh [v2_0|v1_1]"
            exit 0
            ;;
        v2_0) VERSION='v2.0'
            ;;
        v1_1) VERSION='v1.1'
            ;;
        *) echo "Invalid argument $1...exiting"
            exit 0
            ;;
    esac
    shift
done

# Download the SQuAD training and dev datasets
echo "Step 1: Downloading SQuAD-${VERSION} training and dev datasets to ./data/squad"
if [ ! -d "./data" ]; then
    mkdir -p data
else
    echo 'data directory existed'
fi

pushd data
if [ ! -d "./squad" ]; then
    mkdir -p squad
    pushd squad
    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-${VERSION}.json
    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-${VERSION}.json
    popd
else 
    echo 'squad directory existed'
fi

echo "Step 2: Downloading model file and config to ./data/bert-large-uncased"

if [ ! -d "./bert-large-uncased" ]; then
    wget https://drive.google.com/file/d/1eD8QBkbK6YN-_YXODp3tmpp3cZKlrPTA/view?usp=drive_link
    unzip bert-large-uncased.zip -d ./
    rm -f bert-large-uncased.zip
else 
    echo 'bert-large-uncased directory existed'
fi
popd
