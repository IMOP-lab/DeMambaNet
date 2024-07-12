
# DeMambaNet: Deformable Convolution and Mamba Integration Network for High-Precision Segmentation of Ambiguously Defined Dental Radicular Boundaries

# Methods
<div align=center>
  <img src="https://github.com/IMOP-lab/SASAN-Pytorch/blob/main/figures/FINE.png"width=80% height=80%>
</div>
<p align=center>
  Figure 2: Structure of the FINE module.
</p>

# Install
- Compile CUDA operators 
```bash
cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```
- You can also install the operator using .whl files
[DCNv3-1.0-whl](https://github.com/OpenGVLab/InternImage/releases/tag/whl_files)

- For [mamba](https://github.com/state-spaces/mamba): MAMBA-SSM and causal conv1d need to be installed, you can view the original github to install.

- This code uses versions of torch and cuda
```bash
pip install -r requirements.txt
```