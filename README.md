# cs231n

Course assignments of [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/)

## Assignment #1: Image Classification, kNN, SVM, Softmax, Neural Network
Q1: [k-Nearest Neighbor classifier](assignments/assignment1/knn.ipynb)
- Test accuracy on CIFAR-10: 0.282

Q2: [Training a Support Vector Machine](assignments/assignment1/svm.ipynb)
- Test accuracy on CIFAR-10: 0.376

Q3: [Implement a Softmax classifier](assignments/assignment1/softmax.ipynb)
- Test accuracy on CIFAR-10: 0.355

Q4: [Two-Layer Neural Network](assignments/assignment1/two_layer_net.ipynb)
- Test accuracy on CIFAR-10: 0.501

Q5: [Higher Level Representations: Image Features](assignments/assignment1/features.ipynb)
- Test accuracy on CIFAR-10: 0.576

## Assignment #2: Fully-Connected Nets, Batch Normalization, Dropout, Convolutional Nets
Q1: [Fully-connected Neural Network](assignments/assignment2/FullyConnectedNets.ipynb)
- Validation / test accuracy on CIFAR-10: 0.547 / 0.539

Q2: [Batch Normalization](assignments/assignment2/BatchNormalization.ipynb)

Q3: [Dropout](assignments/assignment2/Dropout.ipynb)

Q4: [Convolutional Networks](assignments/assignment2/ConvolutionalNetworks.ipynb)

Q5: [PyTorch](assignments/assignment2/PyTorch.ipynb) / [TensorFlow](assignments/assignment2/TensorFlow.ipynb) on CIFAR-10 ([Tweaked TF model](assignments/assignment2/TensorFlow_my_model.ipynb))
- Training / validation / test accuracy on CIFAR-10: 0.928 / 0.801 / 0.822

| Model       | Training Accuracy | Test Accuracy |
| ----------- |:-----------------:| :------------:|
| Base network | 92.86 | 88.90 |
| VGG-16  | 99.98  | 93.16 |
| VGG-19  | 99.98  | 93.24 |
| ResNet-18  | 99.99  | 93.73 |
| ResNet-101  | 99.99 | 93.76 |

## Assignment #3: Image Captioning with Vanilla RNNs, Image Captioning with LSTMs, Network Visualization, Style Transfer, Generative Adversarial Networks
Q1: [Image Captioning with Vanilla RNNs](assignments/assignment3/RNN_Captioning.ipynb)

Q2: [Image Captioning with LSTMs](assignments/assignment3/LSTM_Captioning.ipynb)

Q3: [Network Visualization: Saliency maps, Class Visualization, and Fooling Images](assignments/assignment3/NetworkVisualization-TensorFlow.ipynb)

Q4: [Style Transfer](assignments/assignment3/StyleTransfer-TensorFlow.ipynb)

Q5: [Generative Adversarial Networks](assignments/assignment3/GANs-TensorFlow.ipynb)
