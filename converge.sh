#!/bin/bash
P=~/miniconda3/envs/tlda/bin/python
D="clipart infograph painting quickdraw real sketch"
for t in $D; do ./run.sh logs/converge_none_to-$t.log $P -u converge.py --target $t; done
for a in $D; do for b in $D; do
    [[ $a < $b ]] || continue
    m=ft_dinov3_$a+${b}_pc50_s0
    [ -f models/$m.pt ] || ./run.sh logs/$m.log $P -u finetune.py --source $a $b --targets $a $b --epochs 5
    for t in $D; do
        [[ $t == $a || $t == $b ]] || ./run.sh logs/converge_${m}_to-$t.log $P -u converge.py --init $m --target $t
    done
done; done
