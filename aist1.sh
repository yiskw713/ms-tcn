python train.py ./result/breakfast/ed-tcn/split1/config.yaml
python train.py ./result/breakfast/ed-tcn/split2/config.yaml
python train.py ./result/breakfast/ed-tcn/split3/config.yaml
python train.py ./result/breakfast/ed-tcn/split4/config.yaml

python eval.py ./result/breakfast/ed-tcn/split1/config.yaml test
python eval.py ./result/breakfast/ed-tcn/split2/config.yaml test
python eval.py ./result/breakfast/ed-tcn/split3/config.yaml test
python eval.py ./result/breakfast/ed-tcn/split4/config.yaml test