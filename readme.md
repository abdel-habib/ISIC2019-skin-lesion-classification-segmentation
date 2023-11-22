tensorboard --logdir=runs

python train.py --train_path "datasets/challenge1/train" --valid_path "datasets/challenge1/val" --experiment_name "SEResnext50_32x4dModel_base_transfer_learning" --network_name "SEResnext50_32x4dModel" --max_epochs "10" --batch_size "32" --verbose "2"