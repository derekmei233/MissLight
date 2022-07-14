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

# Visualization

* For Traffic flow replay
   1. First change directory to **~/CityFlow/frontend**
   2. use browser open index.html file
   3. choose roodnet and replay file from saved directory </br></br>
* If replay not set
  1. Uncomment commented lines in the dqn_generate function in dqn_control script. Then run dqn_control to generate replay files. Or for testing you can run following command under **~/CityFlow/frontend** directory
   - ```python download_replay.py``` </br>
  2. Then following instructions above </br></br>
* To visualize trained data run functions in the **visualization.ipynb**
  - it will be refined in the future.