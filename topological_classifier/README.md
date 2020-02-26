# Topological Classifier

For training our topological classifier, we use pretrained weights from [Places365](). The qualitative results are as follows:

<p align="center">
    <img src="./assets/results_topo_1.png" />
</p>

More details can be found in the paper.

### Requirements

Python 2.7, numpy (1.16.3), torch (1.1.0), torchvision (0.2.2) are bare minimum. For running additional tools like tensorboard, refer to requirements.txt. 

For retraining a model, set model path and uncomment first code block in the main function. The paths to train and test datasets can then be set using the GetDataset object. You have to set the path inside the ImbalancedDatasetSampler class as well.


We have also experimented with many to one and many to many bidirectional LSTMs to leverage the temperal coherency between the frames and the code has been made available in this repo. Dedicated LSTM based classification repo [here](https://github.com/Shubodh/region-classification-cnn-lstm). However, the results have not been reported in the paper as we didn't observe significant improvement for our particular usecase.


