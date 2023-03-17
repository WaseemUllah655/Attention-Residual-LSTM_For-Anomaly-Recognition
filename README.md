# Attention-Residual-LSTM-in-Surveillance-Videos
 Video anomaly recognition in smart cities is an important computer vision task that plays a vital role in smart surveillance and public safety but is challenging due to its diverse, complex, and infrequent occurrence in real-time surveillance environments. Various deep learning models use significant amounts of training data without generalization abilities and with huge time complexity. To overcome these problems, in the current work, we present an efficient light-weight convolutional neural network (CNN)-based anomaly recognition framework that is functional in a surveillance environment with reduced time complexity. We extract spatial CNN features from a series of video frames and feed them to the proposed residual attention-based long short-term memory (LSTM) network, which can precisely recognize anomalous activity in surveillance videos. The representative CNN features with the residual blocks concept in LSTM for sequence learning prove to be effective for anomaly detection and recognition, validating our model’s effective usage in smart cities video surveillance. Extensive experiments on the real-world benchmark UCF-Crime dataset validate the effectiveness of the proposed model within complex surveillance environments and demonstrate that our proposed model outperforms state-of-the-art models with a 1.77%, 0.76%, and 8.62% increase in accuracy on the UCF-Crime, UMN and Avenue datasets, respectively.

# Anomaly recognition 
This work has been published in MDPI sensors journal.
The title of the paper is "An Efficient Anomaly Recognition Framework Using an Attention Residual LSTM in Surveillance Videos"

Required packages

numpy==1.19.1
keras==2.3.1
tensorflow==2.1.0
matplotlib==3.3.1
sklearn==0.23.2
scipy==1.5.2
h5py==2.10.0




@article{ullah2021efficient,
  title={An efficient anomaly recognition framework using an attention residual LSTM in surveillance videos},
  author={Ullah, Waseem and Ullah, Amin and Hussain, Tanveer and Khan, Zulfiqar Ahmad and Baik, Sung Wook},
  journal={Sensors},
  volume={21},
  number={8},
  pages={2811},
  year={2021},
  publisher={MDPI}
}