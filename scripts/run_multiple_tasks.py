import subprocess

datasets = ['davis', 'kiba']
splits = ['random', 'cold_drug', 'cold_target', 'cold_pair']
seeds = [0, 1, 2, 3, 4]
alphas = [0.05, 0.1, 0.2]

for dataset in datasets:
    for split in splits:
        for seed in seeds:
            for alpha in alphas:
                command = [
                    "python", "scripts/train_graphdta_point.py", 
                    "--dataset", dataset, 
                    "--split", split, 
                    "--seed", str(seed), 
                    "--model", "gat_gcn",  
                    "--lr", "5e-4", 
                    "--batch_size", "512", 
                    "--max_epochs", "200", 
                    "--patience", "20", 
                    "--val_frac", "0.1", 
                    "--num_workers", "4", 
                    "--alpha", str(alpha),  
                    "--out_subdir", f"results/tables/{dataset}_{split}_seed{seed}_alpha{alpha}"  
                ]
                subprocess.run(command)
