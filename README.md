

kefu321

rsync -avz /home/ke/code/sim/src/waymo_data/xflow512_matchraw_l1_v1-epoch=27-step=53284-valmeta=0.6499.ckpt ke@10.87.114.128:~/keguo/sim/src/waymo_data/ #full/

rsync -avz -e "ssh -p 32884" ./waymo_data/xflow512_matchraw_l1_v1-epoch=27-step=53284-valmeta=0.6499.ckpt guoke@sprl-server9.dynip.ntu.edu.sg:~/sim/src/waymo_data/ #full/

rsync -avz /home/ke/code/sim/src/waymo_data/xflow512_match_l1_v1-epoch=63-step=60928-valmeta=0.6509.ckpt ke@10.87.225.106:~/code/sim/src/waymo_data/ #full/

rsync -avz ke@10.87.114.128:~/keguo/sim/src/waymo_data/full/training_mapall_init5v ./

rsync -avz ke@10.87.114.128:~/keguo/sim/src/logs/all32_08_match1_05sum_l1sq_w1_branch1_R02/2026-09-05_13-14-16/bc/464l09bs/checkpoints/all32_08_match1_05sum_l1sq_w1_branch1_R02-epoch=3-step=152321-valmeta=0.6490.ckpt ./

rsync -avz ke@10.87.225.106:~/code/sim/src/logs/all32_d58_std005_token10_pre_sq/2026-08-15_21-39-04/bc/iwgs04wt/checkpoints/all32_d58_std005_token10_pre_sq-epoch=3-step=159798-valmeta=0.6539.ckpt ./

rsync -avz -e "ssh -p 32884" guoke@sprl-server9.dynip.ntu.edu.sg:~/sim/src/logs/xflow512_matchraw_l1_v1/2026-09-08_11-16-55/bc/buplip36/checkpoints/xflow512_matchraw_l1_v1-epoch=27-step=53284-valmeta=0.6499.ckpt ./


qsub -I -l select=1:ngpus=1 -l walltime=24:00:00 -P personal-ke.guo
myquota -p personal-ke.guo

Gk@1402862912

source ~/miniconda3/bin/activate
cd ~/scratch/sim/src
conda activate catk

conda create -y -n catk python=3.11.9
conda activate catk
conda install -y -c conda-forge ffmpeg=4.3.2
pip install -r install/requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

sudo apt-get install sumo sumo-tools sumo-doc
pip install -r TrafficManager/requirements.txt

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch_scatter torch_cluster -f  https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install --no-cache-dir --no-deps waymo-open-dataset-tf-2-12-0==1.6.5
pip install -r install/requirements.txt

wsl -d Ubuntu


ssh guoke@sprl-server9.dynip.ntu.edu.sg -p 32884
140286

export PATH=/usr/local/cuda-12.8/bin:$PATH 
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH 
ulimit -n 65535 
source "/home/guoke/miniconda3/bin/activate" 
cd /home/guoke/sim/src 
conda activate sim 
git pull 
setsid nohup torchrun --nproc_per_node=4 -m run trainer=ddp > 1.log 2>&1 &


setsid  nohup torchrun --nproc_per_node=4  -m run trainer=ddp  >  1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2,3 setsid nohup torchrun --nproc_per_node=2 --master_port=29503  -m run trainer=ddp >  23.log 2>&1 & 

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501  -m run1 trainer=ddp >  2.log 2>&1 & 

CUDA_VISIBLE_DEVICES=1 setsid nohup python -m sd.train1 > 1.log 2>&1 &


#0,2,3 1,2,3  -> 0,1, 2

10.100.40.196
ssh 10.87.225.106
source "/home/ke/miniconda3/bin/activate"
cd /home/ke/code/sim/src
conda activate sim
git pull

ssh 10.87.114.128
source "/home/ke/miniconda3/bin/activate"
cd /home/ke/keguo/sim/src
conda activate sim
git pull

nohup python run1.py >  1.log 2>&1 &

nohup python -m sd.train >  1.log 2>&1 &

#to do : continous action, joint distribution by copula , continuous map point,  semi-gradient, IV-learn, interval 0.1 ,, 2048 unknown token


# value network to reject sampling
#counterfact 
#less pt2pt
#non-edge to edge 

#running mean and std of advanatage 

# KL+REVERSE kl

#post sampling to solve goal-conditioned


#aril logpi shape

https://github.com/seolhokim/InverseRL-Pytorch/tree/main
# VAIL 

all2_clean

value use other action

centric discriminator: AIRL64_value0001_disexpertvalidcentric


AIRL64_value0001noclip_distr402060a5_expertvalid influence of range 

python data_preprocess.py
python run.py


1. locate the area or agent where causes  the traffic jam 
2. simulate the bird behavior

#only need keep all thus pred agent all valid , allow new entry agent and exit agent .

#do causal intervention, remove the neighbor 


# also allow high order ,weighted 

#autoregressive entry new agent. 



#diffusion initailization 

To alleviate the imbalance of these two control tokens, we set the label weights:w(<KEEP AGENT>) = 0.1 and w(<REMOVE AGENT>) = 0.9when calculating the CrossEntropy Loss


clean predict nosie v loss


#sim 16-> sim 18->gt gen 



For each
agent, it sequentially predicts (a) an agent type token,
(b) a position token, (c) tokens for size and dynamic
attributes, and (d) a sequence of trajectory tokens.
Each prediction is conditioned on the map, ego history,
agent type specification, and all previously generated
agents.


如果你正在用 LS，可以换成 MaxSup 试试。 与其惩罚 Ground-truth 的 logit，不如直接去压那个最大的 logit。

add spectral norm and softplus activation loss

decomp gail for discriminator
diffusion gail

dit128_airl_128pad64_lambda1_map100_t

WO53LWBDQV
kefu321
todesk：