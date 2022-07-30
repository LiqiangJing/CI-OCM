#!/bin/bash
if [ $# -eq 0 ];
then
    gpuid="0"
    echo "Not gpuid param, default 0"
else
    gpuid=$1
fi
echo "Running on GPU-${gpuid}"

# config
export CUDA_VISIBLE_DEVICES=${gpuid}
nondisjoint="../checkpoints/nondisjoint_best.pt"
disjoint="../checkpoints/disjoint_best.pt"

# job
echo $(date)

echo -e "\n\nRunning... [AUC_FITB nondisjoint]"
python -u  inference_auc_fitb.py  --polyvore-split nondisjoint -test ${nondisjoint}
echo -e "\n\nRunning... [AUC_FITB disjoint]"
python -u  inference_auc_fitb.py  --polyvore-split disjoint -test ${disjoint}

echo -e "\n\nRunning... [Retrieval nondisjoint]"
python -u  inference_retrieval.py  --polyvore-split nondisjoint -test ${nondisjoint}
echo -e "\n\nRunning... [Retrieval disjoint]"
python -u  inference_retrieval.py  --polyvore-split disjoint -test ${disjoint}