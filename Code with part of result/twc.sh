if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/rootfile"
    exit 1
fi

input_file=$1

python3 scripts/TWC_single.py "$input_file"
