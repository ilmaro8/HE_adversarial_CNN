# HE_adversarial_CNN
Implementation of H\&E-adversarial network: a convolutional neural network to learn stain-invariant features through Hematoxylin & Eosin regression.

## Reference
If you find this repository useful in your research, please cite:

[1] Marini, N., Atzori, M., Otálora, S., Marchand-Maillet, S., & Müller, H. H&E-adversarial network: a convolutional neural network to learn stain-invariant features through Hematoxylin & Eosin regression.

Paper link: https://openaccess.thecvf.com/content/ICCV2021W/CDPath/papers/Marini_HE-Adversarial_Network_A_Convolutional_Neural_Network_To_Learn_Stain-Invariant_Features_ICCVW_2021_paper.pdf

## Requirements
Python==3.6.9, albumentations==0.1.8, numpy==1.17.3, opencv==4.2.0, pandas==0.25.2, pillow==6.1.0, torchvision==0.8.1, pytorch==1.7.0

## CSV Input Files:
CSV files are used as input for the scripts. For each partition (train, validation, test), the csv file has path_to_image, class_label as columns.
For prostate experiments, the class_label can be: 
0: benign
1: Gleason pattern 3
2: Gleason pattern 4
3: Gleason pattern 5
For colon experiments, the class_label can be:
0: cancer
1: dysplasia
2: normal glands

## Training
Scripts to train the CNN at path-level, in a fully-supervised fashion.
Some parameters must be manually changed, such as the number of classes (output of the network).

- Training_script.py -n -b -c -e -t -f -i -o. The script is used to train the CNN without any augmentation (no_augment), with colour augmentation (augment).
  * -n: number of the experiment for the training
  * -b: batch size (32)
  * -c: CNN backbone to use (densenet121)
  * -e: number of epochs (10)
  * -t: task of the network (no_augment, augment, normalizer)
  * -f: if True an embedding layer with 128 nodes is inserted before the output layer
  * -i: path of the folder where the input csvs for training (train.csv), validation (valid.csv) and testing (test.csv) are stored
  * -o: path of the folder where to store the CNN’s weights.
  
- Training_stain_augmentation.py -n -b -c -e -t -f -i -o. The script is used to train the CNN with stain augmentation method.
  * -n: number of the experiment for the training
  * -b: batch size (32)
  * -c: CNN backbone to use (densenet121)
  * -e: number of epochs (10)
  * -t: task of the network (no_augment, augment)
  * -f: if True an embedding layer with 128 nodes is inserted before the output layer
  * -i: path of the folder where the input csvs for training (train.csv), validation (valid.csv) and testing (test.csv) are stored
  * -o: path of the folder where to store the CNN’s weights.
- Training_CNN_stain_normalizer.py -n -b -c -e -t -f -i -o. The script is used to train the model using StainCNNs to normalize the images (from https://github.com/alexmomeni/StainGAN).
  * -n: number of the experiment for the training
  * -b: batch size (32)
  * -c: CNN backbone to use (densenet121)
  * -e: number of epochs (10)
  * -t: task of the network (no_augment, augment)
  * -f: if True an embedding layer with 128 nodes is inserted before the output layer
  * -i: path of the folder where the input csvs for training (train.csv), validation (valid.csv) and testing (test.csv) are stored
  * -o: path of the folder where to store the CNN’s weights.
- Training_script_domain_adversarial_center.py -n -b -c -e -t -f -i -o.  The script is used to train the CNN using a domain adversarial CNN. Data generator must be modified, inserting the condition to distinguish the images from different domains (in this paper the path). 
  * -n: number of the experiment for the training
  * -b: batch size (32)
  * -c: CNN backbone to use (densenet121)
  * -e: number of epochs (10)
  * -t: task of the network (domain_center)
  * -f: if True an embedding layer with 128 nodes is inserted before the output layer
  * -i: path of the folder where the input csvs for training (train.csv), validation (valid.csv) and testing (test.csv) are stored
  * -o: path of the folder where to store the CNN’s weights.
- Training_script_domain_adversarial_regressor.py -n -b -c -e -t -f -i -o. The script is used to train H&E-adversarial network.
  * -n: number of the experiment for the training
  * -b: batch size (32)
  * -c: CNN backbone to use (densenet121)
  * -e: number of epochs (10)
  * -t: task of the network (domain_regressor)
  * -f: if True an embedding layer with 128 nodes is inserted before the output layer
  * -i: path of the folder where the input csvs for training (train.csv), validation (valid.csv) and testing (test.csv) are stored
  * -o: path of the folder where to store the CNN’s weights.


## Testing
Scripts to test the CNNs.

- Testing_script.py -n -b -c -t -f -d -m -i. The script is used to test the CNN trained using no_augment, augment, stain_augment
  * -n: number of the experiment for the training
  * -b: batch size
  * -c: CNN backbone to use (densenet121)
  * -t: task of the network (no_augment, augment, stain_augment, normalizer). In case the task is “normalizer” the image used as target must be the same used during the  training.
  * -f: if True an embedding layer with 128 nodes is inserted before the output layer. It must be the same used to train the model
  * -d: name of the dataset to use.
  * -m: path where the model is stored.
  * -i: path where the csv input files are stored.
  
- Testing_script_CNN_normalized.py. -n -b -c -t -f -d -m -i. The script is used to test the CNN trained using StainCNNs to normalize the images.
  * -n: number of the experiment for the training
  * -b: batch size
  * -c: CNN backbone to use (densenet121)
  * -t: task of the network (StainGAN, StainNet)
  * -f: if True an embedding layer with 128 nodes is inserted before the output layer. It must be the same used to train the model
  * -d: name of the dataset to use.
  * -m: path where the model is stored.
  * -i: path where the csv input files are stored.

- Testing_script_domain_adversarial_center.py. -n -b -c -t -f -d -m -i. The script is used to test the CNN trained using domain_adversarial network
  * -n: number of the experiment for the training
  * -b: batch size
  * -c: CNN backbone to use (densenet121)
  * -t: task of the network (domain_center)
  * -f: if True an embedding layer with 128 nodes is inserted before the output layer. It must be the same used to train the model
  * -d: name of the dataset to use.
  * -m: path where the model is stored.
  * -i: path where the csv input files are stored.

- Testing_script_domain_adversarial_regressor.py. -n -b -c -t -f -d -m -i. The script is used to test the CNN trained using no_augment, augment, stain_augment
  * -n: number of the experiment for the training
  * -b: batch size
  * -c: CNN backbone to use (densenet121)
  * -t: task of the network (domain_regressor)
  * -f: if True an embedding layer with 128 nodes is inserted before the output layer. It must be the same used to train the model
  * -d: name of the dataset to use.
  * -m: path where the model is stored.
  * -i: path where the csv input files are stored.

## Acknoledgements
This project has received funding from the EuropeanUnion’s Horizon 2020 research and innovation programme under grant agree-ment No. 825292 [ExaMode](http://www.examode.eu). Infrastructure fromthe SURFsara HPC center was used to train the CNN models in parallel. 
