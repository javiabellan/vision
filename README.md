<h1 align="center">Vision</h1>

> - [Pretrained models in pytorch](https://github.com/Cadene/pretrained-models.pytorch)
> - [Ranking](https://openreview.net/pdf?id=Hyzq4ZKa97),
> - [comparison paper](https://arxiv.org/pdf/1810.00736.pdf)
> - [Little tricks paper](https://arxiv.org/abs/1812.01187)
> - [GPipe](https://arxiv.org/pdf/1811.06965v4.pdf)
- [Convolutional Neural Network (CNN)](/posts/5-vision/cnn.md) For fixed size oredered data, like images
  - Variable input size: use **adaptative pooling**, final layers then:
    - Option 1: `AdaptiveAvgPool2d((1, 1))` -> `Linear(num_features, num_classes)` (less computation)
    - Option 2: `Conv2d(num_features, num_classes, 3, padding=1)` -> `AdaptiveAvgPool2d((1, 1))`
- **Small nets**: Useful for mobile phones.
  - **SqueezeNet** (2016): v1.0: `58.108`, v1.1: `58.250`. [*paper*](https://arxiv.org/abs/1602.07360).
  - **Mobilenet v1** (2017): `69.600`The standard convolution is decomposed into two. Accuracy similar to Resnet-18. [*paper*](https://arxiv.org/abs/1704.04861)
  - **Shufflenet** (2017): The most efficient net `67.400`. [*paper*](https://arxiv.org/abs/1707.01083).
  - **NASNet-A-Mobile** (2017): `74.080`. [*paper*](https://arxiv.org/abs/1707.07012)
  - **Mobilenet v2** (2018): `71.800`. [*paper*](https://arxiv.org/abs/1801.04381)
  - **SqueezeNext** (2018): `62.640`. Hardware-Aware Neural network design. [*paper*](https://arxiv.org/abs/1803.10615).
- **Common nets**:
  - **Inception v3** (2015): `77.294` [*paper*](https://arxiv.org/abs/1512.00567), [*blog*](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202)
  - **Resnet** (2015): Every 2 convolutions (3x3->3x3) **sum** the original input. [*paper*](https://arxiv.org/abs/1512.03385) Wide ResNet?
    - **Resnet-18**: `70.142`
    - **Resnet-34**: `73.554`
    - **Resnet-50**: `76.002`. **SE-ResNet50**: `77.636`. **SE-ResNeXt50 (32x4d)**: `79.076`
    - **Resnet-101**: `77.438`. **SE-ResNet101**: `78.396`. **SE-ResNeXt101 (32x4d)**: `80.236`
    - **Resnet-152**: `78.428`. **SE-ResNet152**: `78.658`
  - **Densenet** (2016): Every 2 convolutions (3x3->1x1) **concatenate** the original input. [*paper*](https://arxiv.org/abs/1608.06993)
    - **DenseNet-121**: `74.646`
    - **DenseNet-169**: `76.026`
    - **DenseNet-201**: `77.152`
    - **DenseNet-161**: `77.560`
  - **Xception** (2016): `78.888` [*paper*](https://arxiv.org/abs/1610.02357)
  - **ResNext** (2016): [*paper*](https://arxiv.org/abs/1611.05431)
    - **ResNeXt101 (32x4d)**:	`78.188`
    - **ResNeXt101 (64x4d)**:	`78.956`
  - **Dual Path Network (DPN)**: [*paper*](https://arxiv.org/abs/1707.01629)
    - **DualPathNet98**: `79.224`
    - **DualPathNet92_5k**: `79.400`
    - **DualPathNet131**: `79.432`
    - **DualPathNet107_5k**: `79.746`
  - **SENet** (2017): Squeeze and Excitation network. Net is allowed to adaptively adjust the weighting of each feature map in the convolution block. [*paper*](https://arxiv.org/abs/1709.01507)
    - **SE-ResNet50**: `77.636`
    - **SE-ResNet101**: `78.396`
    - **SE-ResNet152**: `78.658`
    - **SE-ResNeXt50 (32x4d)**: `79.076` **USE THIS ONE FOR A MEDIUM NET**
    - **SE-ResNeXt101 (32x4d)**: `80.236` **USE THIS ONE FOR A BIG NET**
- **Giants nets**: Useful for competitions.
  - **Inception v4**: `80.062`, **Inception-ResNet**: `80.170` [*paper*](https://arxiv.org/abs/1602.07261)
  - **PolyNet**: `81.002`
  - **SENet-154**: `81.304`
  - **NASNet-A-Large**: `82.566` Crated with AutoML. [*paper*](https://arxiv.org/abs/1707.07012)
  - **PNASNet-5-Large**: `82.736`
  - **AmoebaNet**: `83.000` [*paper*](https://arxiv.org/abs/1802.01548)


### [5.2 CNN Black box explanation](/posts/5-vision/explanation.md)
[*link 1*](https://github.com/utkuozbulak/pytorch-cnn-visualizations), [*link 2*](https://ramprs.github.io/2017/01/21/Grad-CAM-Making-Off-the-Shelf-Deep-Models-Transparent-through-Visual-Explanations.html)
- **Features**: Average features on the channel axis. This shows all classes detected. `[512, 11, 11]-->[11, 11]`.
- **CAM**: Class Activation Map. Final features multiplied by a single class weights and then averaged. `[512, 11, 11]*[512]-->[11, 11]`. [*paper*](https://arxiv.org/abs/1512.04150).
- **Grad-CAM**: Final features multiplied by class gradients and the averaged. [*paper*](https://arxiv.org/abs/1610.02391).
- **SmoothGrad** [*paper*](https://arxiv.org/abs/1706.03825).
- Extra: [Distill: feature visualization](https://distill.pub/2017/feature-visualization/)
- Extra: [Distill: building blocks](https://distill.pub/2018/building-blocks/)

### [5.3 Object detection](/posts/5-vision/detection.md)
Get bounding boxes. Check [**detectron**](https://github.com/facebookresearch/detectron).
- Region-based methods
  - **R-CNN**:        [*paper*](https://arxiv.org/abs/1311.2524)
  - **Fast R-CNN**:   [*paper*](https://arxiv.org/abs/1504.08083)
  - **Faster R-CNN**: [*paper*](https://arxiv.org/abs/1506.01497)
  - **Mask R-CNN**: [*paper*](https://arxiv.org/abs/1703.06870)
- Single-shot methods
  - **YOLOv1**: [*paper*](https://arxiv.org/abs/1506.02640)
  - **SSD**:    [*paper*](https://arxiv.org/abs/1512.02325)
  - **YOLOv2**: [*paper*](https://arxiv.org/abs/1612.08242)
  - **YOLOv3**: [*paper*](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
  - **Feature Pyramid Networks (FPN)** (2016) [*paper*](https://arxiv.org/abs/1612.03144), [*slides*](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
  - **RetinaNet**: (2017) Focal Loss for Dense Object Detection [*paper*](https://arxiv.org/abs/1708.02002)
  - **Path Aggregation Network** (2018) [*paper*](https://arxiv.org/abs/1803.01534)

### [5.4 Segmentation](/posts/5-vision/segmentation.md)
Get pixel-level classes. Note that the model backbone can be a resnet, densenet, inception... Check [this repo](https://github.com/qubvel/segmentation_models)

- Semantic segmentation
  - **FCN** Fully Convolutional Networks (2014)
  - **SegNet** (2015)
  - **Unet**: Concatenate like a densenet [facebook](https://github.com/facebookresearch/fastMRI/tree/master/models/unet).
  - **LinkNet**: Adds like a resnet [*paper*](https://arxiv.org/abs/1707.03718)
  - **Pyramid Scene Parsing Net (PSPNet)** [*paper*](https://arxiv.org/abs/1612.01105)
  - **DeepLabv3** SotA. Increasing dilatation, increases field-of-view. [paper](https://arxiv.org/abs/1706.05587)
- Instance segmentation
  - **Feature Pyramid Networks (FPN)** (2016) [*paper*](https://arxiv.org/abs/1612.03144), [*slides*](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
  - **Path Aggregation Network** (2018) [*paper*](https://arxiv.org/abs/1803.01534)
  
### [5.5 Generative](/posts/5-vision/generative.md)
Useful for data augmentation, B&W colorization, super-resolution, artistic style...

- **No GANs** (Image-to-image):
  - **Model**: Pretrained Unet 
  - **Loss functions**:
     - **Pixel MSE**: Flat the 2D images and compare them with regular MSE.
     - **Discriminator/Critic** The loss function is a binary classification pretrained resnet (real/fake).
     - **Feature losses** or perpetual losses.
  - pix2pixHD
  - COVST: Naively add temporal consistency.
  - [Video-to-Video Synthesis](https://tcwang0509.github.io/vid2vid/)
- [Generative advesarial network (GAN)](/posts/5-vision/gan.md)
  - Process
    1. Train a bit the generator and save generated images. `unet_learner` with pixelMSE loss
    2. Train bit the discriminator with real vs generated images. `create_critic_learner`
    3. Ping-pong train both nets `GANLearner` with 2 losses pixelMSE and discriminator.
  - Discriminative model with Spectral Normalization
  - Loss with adaptive loss
  - Metric accuracy is accuracy_thres_expand
  - [infoGAN](http://www.depthfirstlearning.com/2018/InfoGAN)
  - BigGAN: SotA in image synthesis. Same GAN techiques, much larger scale. Increase model capacity + increase batch size.
  - [10 types of GANs](https://amp.reddit.com/r/MachineLearning/comments/8z97mx/r_math_insights_from_10_gan_papers_infogans)
- [Inceptionism](/posts/5-vision/inceptionism.md)
- [Capsule net](/posts/5-vision/capsule.md)

> To speed up jpeg image I/O from the disk one should not use PIL, skimage and even OpenCV but look for libjpeg-turbo or PyVips.
