# To run this program

## Configure environment

### Dependencies
```
pip install -r requirements.txt
```

### CityFlow
```
git clone https://github.com/cityflow-project/CityFlow.git
pip install .
```


## Run experiments

```
python run.py
```

## Change experiment settings
1. change --config to 'syn4x4, hz4x4, ny16x3' to navigate different datasets

2. choose -control from ['F-F','I-F','I-M','M-M','S-S-A','S-S-O', 'I-I', 'S-S-O-model_based'] to run different approaches in paper.

3. change --prefix to distinguish data storage locations

4. --debug to keep replay of cityflow simulation

5. assign --mask_pos to set the missing intersection index

## Notice:

this repository is only for reproduce results in the paper, later version will be released soon based on LibSignal project with compacted and flexible formation.

For your reference:
Paper link:

