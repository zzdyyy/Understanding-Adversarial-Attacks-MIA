export CUDA_VISIBLE_DEVICES=0
for attack in cw-li; do
    for eps in 0.{1..9..2}; do
        python craft_adv_examples.py -d derm -a $attack -b 100 -e $eps -c 100
    done
done
for attack in cw-li; do
    for eps in 0.{1..9..2}; do
        python craft_adv_examples.py -d cxr -a $attack -b 100 -e $eps -c 100
    done
done

CUDA_VISIBLE_DEVICES=0 python craft_adv_examples.py -d derm -a cw-li -b 64 -e 1 -c 100
CUDA_VISIBLE_DEVICES=0 python craft_adv_examples.py -d cxr -a cw-li -b 64 -e 1 -c 100
CUDA_VISIBLE_DEVICES=1 python craft_adv_examples.py -d dr -a cw-li -b 64 -e 1 -c 100

export CUDA_VISIBLE_DEVICES=1
for attack in cw-li; do
    for eps in 0.{1..9..2}; do
        python craft_adv_examples.py -d dr -a $attack -b 100 -e $eps -c 100
    done
done

http://bwc.buaa.edu.cn/info/1026/1741.htm


export CUDA_VISIBLE_DEVICES=0
for ds in derm cxr; do
    for attack in pgd pgd_bb ; do
        echo python extract_features.py -d $ds -a $attack -b 100 >> log/history.log
        python extract_features.py -d $ds -a $attack -b 100
        python extract_characteristics.py -d $ds -a $attack -b 100 -r lid
        python extract_characteristics.py -d $ds -a $attack -b 100 -r bu
        python extract_characteristics.py -d $ds -a $attack -b 100 -r kd
    done
done

export CUDA_VISIBLE_DEVICES=0
for ds in cxr; do
    for attack in bim bim_bb pgd pgd_bb ; do
        python extract_characteristics.py -d $ds -a $attack -b 100 -r kd
    done
done


export CUDA_VISIBLE_DEVICES=1
for ds in cxr; do
    for attack in fgsm deepfool deepfool_bb; do
        python extract_characteristics.py -d $ds -a $attack -b 100 -r kd
    done
done


for ds in dr; do
    for attack in pgd pgd_bb ; do
        echo python extract_features.py -d $ds -a $attack -b 100 >> log/history.log
        python extract_features.py -d $ds -a $attack -b 100
        python extract_characteristics.py -d $ds -a $attack -b 100 -r lid
        python extract_characteristics.py -d $ds -a $attack -b 100 -r bu
        python extract_characteristics.py -d $ds -a $attack -b 100 -r kd
    done
done



export CUDA_VISIBLE_DEVICES=1
for ds in dr; do
    for attack in cw-li ; do
        echo python extract_features.py -d $ds -a $attack -b 60 -c 100 >> log/history.log
        python extract_features.py -d $ds -a $attack -b 60 -c 100
        python extract_characteristics.py -d $ds -a $attack -b 60 -r lid -c 100
        python extract_characteristics.py -d $ds -a $attack -b 60 -r bu -c 100
        python extract_characteristics.py -d $ds -a $attack -b 60 -r kd -c 100
    done
done


export CUDA_VISIBLE_DEVICES=1
for dataset in cxr05 cxr056 cxr0456; do
    for attack in fgsm; do
        for eps in 0.{1..9..2} {1..10}; do
            python craft_adv_examples.py -d $dataset -a $attack -b 100 -e $eps
        done
    done
    for attack in bim pgd; do
        for eps in 0.{1..9} 1.0; do
            python craft_adv_examples.py -d $dataset -a $attack -b 100 -e $eps
        done
    done
done

export CUDA_VISIBLE_DEVICES=1
for dataset in cxr05 cxr056 cxr0456; do
    for attack in cw-li; do
        for eps in 0.{1..9} 1.0; do
            python craft_adv_examples.py -d $dataset -a $attack -c 100 -b 100 -e $eps
        done
    done
done