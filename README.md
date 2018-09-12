# Style Transfer with Neural Network
## Introduction
Task: Given an image (style image), transfer its stylistic elements to other images (content images) while keeping their semantics.
## Approach
The content images are passed through a transformation neural network. This neural network is trained so that the transformed images match the style of the style image (through the use of a style loss derived from Gramian matrix) while retaining its content (through the use of a content loss).
The style- and content-matching is not done at pixel space, but on feature representations of the images, obtained using a pretrained VGG-19 neural network.

## Demo:
### Content:

![Content](https://github.com/nhatsmrt/NeuralStyleTransfer/blob/CycleGAN/Predictions/la_muse.jpg)


### Style:
![Style](https://github.com/nhatsmrt/NeuralStyleTransfer/blob/CycleGAN/mouse.png)


### Result:

![Result](https://github.com/nhatsmrt/NeuralStyleTransfer/blob/CycleGAN/Predictions/la_muse_styled.png)
