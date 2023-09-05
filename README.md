# TransMatch
Pytorch implementation of Paper "TransMatch: Transformer-based Correspondence Pruning via Local and Global Consensus" 
## Requirements
Please use Python 3.6, opencv-contrib-python (3.4.0.12) and Pytorch (>= 1.1.0). Other dependencies should be easily installed through pip or conda.
# Preparing Data
Please follow their instructions to download the training and testing data.
```
bash download_data.sh raw_data raw_data_yfcc.tar.gz 0 8 ## YFCC100M
tar -xvf raw_data_yfcc.tar.gz

bash download_data.sh raw_sun3d_test raw_sun3d_test.tar.gz 0 2 ## SUN3D
tar -xvf raw_sun3d_test.tar.gz
bash download_data.sh raw_sun3d_train raw_sun3d_train.tar.gz 0 63
tar -xvf raw_sun3d_train.tar.gz
```
After downloading the datasets, the initial matches for YFCC100M and SUN3D can be generated as follows: Here we provide descriptors for SIFT (default)
```
cd dump_match
python extract_feature.py
python yfcc.py
python extract_feature.py --input_path=../raw_data/sun3d_test
python sun3d.py
```
# Testing and Training Model
We provide a pretrained model on YFCC100M. The results in our paper can be reproduced by running the test script:
```
python main.py --run_mode=test --model_path=../model/yfcc
```
Set ```--use_ransac=True``` to get results after RANSAC post-processing.
if you want to retrain the model on YFCC100M, run the training script.
```
python main.py
```
You can also retrain the model on SUN3D by changing ```--data_tr```, ```--data_va```, ```--data_te``` in config.py
