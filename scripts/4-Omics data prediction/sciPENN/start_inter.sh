
#!/usr/bin/env sh
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

if [ "$#" -eq 0 ]; then
    echo "Error: Please pass at least one dataset pair, e.g.: sh start.sh \"dataset1 dataset2\" \"dataset3 dataset1\""
    exit 1
fi

start=1
repeat_len=1

for pair in "$@"; do
    read -r dataset1 dataset2 << EOF
$pair
EOF

    if [ -z "$dataset1" ] || [ -z "$dataset2" ]; then
        echo "Error: Invalid dataset pair format: $pair"
        exit 1
    fi

    n1=$(echo "$dataset1" | sed 's/[^0-9]//g')
    n2=$(echo "$dataset2" | sed 's/[^0-9]//g')

    if [ "$n1" -le "$n2" ]; then
        pair_dir="${dataset1}_${dataset2}"
    else
        pair_dir="${dataset2}_${dataset1}"
    fi

    RNA_path_train="/home/jyx/DePass-main/outputs/DePassData/${pair_dir}/${dataset1}/adata_RNA.h5ad"
    Pro_path_train="/home/jyx/DePass-main/outputs/DePassData/${pair_dir}/${dataset1}/adata_protein.h5ad"

    RNA_path_test="/home/jyx/DePass-main/outputs/DePassData/${pair_dir}/${dataset2}/adata_RNA.h5ad"
    Pro_path_test="/home/jyx/DePass-main/outputs/DePassData/${pair_dir}/${dataset2}/adata_protein.h5ad"

    for r in $(seq "$start" "$repeat_len"); do
        echo "Processing dataset pair: train: $dataset1, test: $dataset2, repeat $r"
        python /home/jyx/depassfortranslation/methods/sc/rna_pro/sciPENN/sciPENN_inter.py \
            --repeat="$r" \
            --Pro_path_train="$Pro_path_train" \
            --RNA_path_train="$RNA_path_train" \
            --Pro_path_test="$Pro_path_test" \
            --RNA_path_test="$RNA_path_test"
    done
done
      
        