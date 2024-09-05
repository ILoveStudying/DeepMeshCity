# DeepMeshCity: A Deep Learning Model for Urban Grid Prediction
Urban grid prediction is a general spatial-temporal forecasting framework in which the urban is partitioned into equally-sized grids and each grid represents the physical quantities of interest that change over time. It takes as input multi-step historical grid observations and produces a next-step prediction over all grids. Many classic spatial-temporal prediction tasks fall into this framework, e.g., air quality prediction, crowd density and traffic flow prediction, and taxi demand forecasting.

# Dataset
we evaluate the ability of DeepMeshCity for urban grid prediction on two typical tasks: crowd density prediction and flow prediction. The experiments are conducted on four real-world urban datasets---BousaiTYO and BousaiOSA crowd density datasets, BousaiTYO crowd flow dataset, and TaxiBJ traffic flow dataset.

TaxiBJ is one of the most widely used traffic flow datasets in the literature The dataset records the GPS coordinates of taxicab in Beijing during four time periods, $07/01/2013-10/30/2013, 03/01/2014/-06/30/2014, 03/01/2015-06/30/2015$, and $11/01/2015-04/10/2016$. The sampling time interval is 30 minutes and the entire period covers 18 months. We partition the city into $32 \times 32$ grid cells, which yields a tensor of size $20016 \times 32 \times 32 \times 2$.

Bousai datasets are provided by Yahoo Japan Corporation. The dataset records the location information of millions of users in Japan with a sampling interval of 30 minutes. The records of two big cities (Tokyo and Osaka) from 1 April 2017 to 9 July 2017 (100 days) are selected in our experiments. We refer to the corresponding datasets as BousaiTYO and BousaiOSA, respectively. The two cities are partitioned into $80 \times 80$ and $60 \times 60$ grid cells, respectively, with a grid cell size $450\text{m} \times 450\text{m}$. Consequently, the BousaiTYO dataset contains a crowd density tensor of size $4800 \times 80 \times 80 \times 1$ and a crowd flow tensor of size $4800 \times 80 \times 80 \times 2$, whereas the BousaiOSA dataset only contains a crowd density tensor of size $4800 \times 60 \times 60 \times 1$.

<div align=center>
<img src="https://github.com/ILoveStudying/DeepMeshCity/blob/master/prediction.png" width="50%" height="50%" > </div> 

# Architecture
The overall architecture of DeepMeshCity is depicted in the following figure. The proposed model has at its core a stack of SA-CGL (Self-Attention Citywide Grid Learner) blocks, which are designed to better handle the global spatial dependencies and the multi-scale spatial-temporal correlations.
 <div align=center>
<img src="https://github.com/ILoveStudying/DeepMeshCity/blob/master/framework.png" width="80%" height="80%" > </div>

# Usage
1. Download data. The Bousai datasets requires the permission from [ Yahoo! Japan](https://github.com/deepkashiwa20/DeepCrowd). However, [TaxiBJ](https://pan.baidu.com/s/1tGQRs5b4kXVkWpwo3WtoBA) can be directly obtained with code **u12b** .
2. Only TaxiBJ requires preprocessing:

```python
cd Data/
python taxibj.py
```
3. Train/Test your model. You can refer to following configs for different dataset. 
```python
python run.py DeepMeshCity_tokyo_density_config.py tokyo density train/test

python run.py DeepMeshCity_tokyo_flow_config.py tokyo flow train/test

python run.py DeepMeshCity_osaka_density_config.py osaka density train/test

python run.py DeepMeshCity_taxibj_flow_config.py taxibj flow train/test
```
# Citation
> Chi Zhang, Linhao Cai, Meng Chen, Xiucheng Li, and Gao Cong. 2024. DeepMeshCity: A Deep Learning Model for Urban Grid Prediction. ACM Trans. Knowl. Discov. Data 18, 6, Article 148 (July 2024), 26 pages. https://doi.org/10.1145/3652859

```
@article{10.1145/3652859,
author = {Zhang, Chi and Cai, Linhao and Chen, Meng and Li, Xiucheng and Cong, Gao},
title = {DeepMeshCity: A Deep Learning Model for Urban Grid Prediction},
year = {2024},
issue_date = {July 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {18},
number = {6},
issn = {1556-4681},
url = {https://doi.org/10.1145/3652859},
doi = {10.1145/3652859},
abstract = {Urban grid prediction can be applied to many classic spatial-temporal prediction tasks such as air quality prediction, crowd density prediction, and traffic flow prediction, which is of great importance to smart city building. In light of its practical values, many methods have been developed for it and have achieved promising results. Despite their successes, two main challenges remain open: (a) how to well capture the global dependencies and (b) how to effectively model the multi-scale spatial-temporal correlations? To address these two challenges, we propose a novel method—DeepMeshCity, with a carefully-designed Self-Attention Citywide Grid Learner (SA-CGL) block comprising of a self-attention unit and a Citywide Grid Learner (CGL) unit. The self-attention block aims to capture the global spatial dependencies, and the CGL unit is responsible for learning the spatial-temporal correlations. In particular, a multi-scale memory unit is proposed to traverse all stacked SA-CGL blocks along a zigzag path to capture the multi-scale spatial-temporal correlations. In addition, we propose to initialize the single-scale memory units and the multi-scale memory units by using the corresponding ones in the previous fragment stack, so as to speed up the model training. We evaluate the performance of our proposed model by comparing with several state-of-the-art methods on four real-world datasets for two urban grid prediction applications. The experimental results verify the superiority of DeepMeshCity over the existing ones. The code is available at https://github.com/ILoveStudying/DeepMeshCity.},
journal = {ACM Trans. Knowl. Discov. Data},
month = {apr},
articleno = {148},
numpages = {26},
keywords = {Spatial-temporal prediction, crowd/traffic flow prediction, urban computing}
}
```
