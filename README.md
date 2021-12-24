# Road Graph
|File name|Note|
|--|--|
rawstate_4x4.pkl |data collected from rl model |
roadnet_relation_4x4.pkl |relation data of intersections and roads |
state_4x4_r1_d0_w0_astgcn|1.generated data including train,val,test;2.flag of road node update|
state_4x4.pkl|transform data's format from raw data to astgcn model's data |

# Configurations
1. For each roadnetwork experiment, you should change: **run_simple_dqn.py** :save_dir,log_dir,state_dir,state_name,relation_name.  
2. The settings for each experiments are given in the "configurations" folder.

# Run Code
1. generate roadnet_relation.pkl and rawstate.pkl 
```
python run_simple_dqn.py --config cityflow_hz_4x4.cfg
```

2. generate train,val,test data which prediction model needed
```
python prepareData.py --config configurations/HZ_4x4_astgcn.conf
```
3. train and test prediction model
```
python train_ASTGCN_r.py --config configurations/HZ_4x4_astgcn.conf
```
