# Easypredict
New code base for temporary pre experiments

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

