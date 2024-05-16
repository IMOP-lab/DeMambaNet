import subprocess

fold = 0
gpu = 0
tr_size = 100000


dataset = "mamba_fre"
output_dir = "mamba_fre"

dataset_folder ="/home/zbf/teech/train/"
subprocess.call(["python", "scripts/main_feat_seg_bl.py",
                 "--src_dir", dataset_folder,
                 "--data_dir", dataset_folder+"/image/",
                 "--save_dir", "./"+output_dir,
                 "--b", "2",
                 "--dataset", dataset,
                 "--epochs", "150",
                 "--gpu", str(gpu),
                 "--fold", str(fold),
                 "--tr_size", str(tr_size),
                 "--num_classes", "2"])
