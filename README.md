# Roadgraph
|File name|Note|
|--|--|
rawstate_4x4.pkl |data collected from rl model |
roadnet_relation_4x4.pkl |relation data of intersections and roads |
state_4x4_r1_d0_w0_astgcn|1.generated data including train,val,test;2.flag of road node update|
state_4x4.pkl|transform data's format from raw data to astgcn model's data |

# Run Code
1. generate roadnet_relation.pkl and rawstate.pkl 
```
run run_simple_dqn.py
```

2. generate train,val,test data which prediction model needed
```
python prepareData.py --config configurations/HZ_4x4_astgcn.conf
```
3. train and test prediction model
```
python train_ASTGCN_r.py --config configurations/HZ_4x4_astgcn.conf
```
