import subprocess

fold = 0
gpu = 0
tr_size = 10
model_type = "vit_b"
output_dir = "61"

model_dir = "pth/61mamba_fremodel.pth"

dataset_folder = "/home/zbf/teech/test"


command = ["python", "scripts/main_autosam_seg_test.py",
           "--src_dir", dataset_folder,
           "--data_dir", f"{dataset_folder}/image/",
           "--save_dir", f"./{output_dir}",
           "--model_dir", f"./{model_dir}",
           "--b", "1",
           # "--dataset", dataset,
           "--gpu", str(gpu),
           "--fold", str(fold),
           "--tr_size", str(tr_size),
           "--model_type", model_type,
           "--num_classes", "2"]

subprocess.call(command)
