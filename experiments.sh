#!/bin/bash
P=~/miniconda3/envs/tlda/bin/python
D="clipart infograph painting quickdraw real sketch"
done_() { grep -qs '"final"' results/$1.json; }
for s in $D; do done_ scratch_resnet50_$s || ./run.sh logs/scratch_resnet50_$s.log $P -u train_from_scratch.py --arch resnet50 --source $s --size 224; done
for s in $D; do done_ pre_resnet50_$s || ./run.sh logs/pre_resnet50_$s.log $P -u train_from_scratch.py --arch resnet50 --source $s --size 224 --pretrained --lr 1e-4 --epochs 10; done
for s in infograph painting quickdraw real; do
    [ -f models/ft_dinov3_${s}_pc0_s0.pt ] || ./run.sh logs/ft_dinov3_${s}_pc0.log $P -u finetune.py --source $s --per-class 0
    done_ ft_dinov3_${s}_pc0_s0 || ./run.sh logs/eval_dinov3_${s}_pc0.log $P -u evaluate.py ft_dinov3_${s}_pc0_s0
done
