# Optical Flow
### **Build environment**
Experiment Environment   
OS: Windows 10  
GPU: NVIDIA GeForce GTX 1050  


```shell
conda create --name 3dcvfinal python=3.8 
conda activate 3dcvfinal
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts 
pip install -r requirements.txt 
```

### **How to run code**
#### **output of program**
There will be 2 outputs of each execution, expx_relative.csv and expx_absolute.csv. Each row of expx_relative.csv is the displacement between two frames. Each row of expx_absolute.csv is the displacement relative to first frame.
#### **Experiment 1** 
```shell
python optical_flow.py --exp 1
```  
#### **Experiment 2** 
```shell
python optical_flow.py --exp 2
```  