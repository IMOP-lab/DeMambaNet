import subprocess

fold = 0
gpu = 0
tr_size = 10
model_type = "vit_b"

output_dir = "pretagain"


model_dir = "pth/41mamba_fremodel.pth"

dataset_folder = "/home/zbf/Desktop/code/teech_mamba/again"


command = ["python", "scripts/two_test.py",
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
