
## What's in this repository
You have reached the repository containing code to reproduce the experiments in our paper:
```
Test Time Transform Prediction for Open Set Histopathological Image Recognition
Adrian Galdran, Katherine Jane Hewitt, Narmin Ghaffari Laleh, Jakob Kather, Gustavo Carneiro, Miguel A. González Ballester
MICCAI - Medical Image Computing and Computer Assisted Interventions 2022
```
Link: [here](https://arxiv.org/abs/2206.10033)


## Video Presentation
You can find watch a 5-minute video presentation we prepared also for MICCAI workshop by clicking in the image below:

<a href="https://www.youtube.com/watch?v=Dt9uAvgPPak">
<p align="center">
<img href="InstantDL" src="other/T3PO.png" width="500" alt="Link to presentation" align="center">
</p>
</a>

## Prepare the data
Please follow these instructions to get the data ready:

* Kather 2016 [link to data source](https://www.nature.com/articles/srep27988)

```
wget https://zenodo.org/record/53169/files/Kather_texture_2016_image_tiles_5000.zip
# below, optionally specify different path to store data, "-d other_path/"
unzip Kather_texture_2016_image_tiles_5000.zip -d data/Kather_texture_2016_image_tiles_5000  
# then you need to use here that path, adding "--path_data_in other_path/Kather_texture_2016_image_tiles_5000"
python data/kather2016.py --path_data_in data/Kather_texture_2016_image_tiles_5000           
rm -r data/Kather_texture_2016_image_tiles_5000
rm Kather_texture_2016_image_tiles_5000.zip
```

* Kather 100k [link to data source](https://doi.org/10.1371/journal.pmed.1002730)
```
wget https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip  
# below, optionally specify different path to store data, "-d other_path/"
unzip NCT-CRC-HE-100K.zip -d data/
# then you need to use here that path, adding "--path_data_in other_path/NCT-CRC-HE-100K"                                
python data/kather100k.py --path_data_in data/NCT-CRC-HE-100K     
rm -r data/NCT-CRC-HE-100K
rm NCT-CRC-HE-100K.zip
```

## Train the models
Afterwards, have a look at `osr_train.sh`, where you can find the instruction to train a model as in our paper. 
Note that testing also happen in the training script, at the end, and results are already logged in a txt file.

## Acknowledgement
Acknowledgement: The code in this repository is very much based on [this codebase](https://github.com/sgvaze/osr_closed_set_all_you_need). In fact, you may find some pieces of code here and there that do nothing, those are parts of the original code that I did not get to remove or clean up, sorry about that.
