# Coronary Artery Segmentation
透過 3D Unet 將心臟的冠狀動脈(包含 左冠狀動脈、有冠狀動脈、左迴旋支、左前降支、右後降支等) 分割出來，其他的後續應用及其研究是秘密歐。

Dataset 來自`聯新醫院`，並自己 Label 製作而成。
## Model
[3D Unet介紹](https://martin12345m.medium.com/3d-unet%E5%B0%8F%E7%B0%A1%E4%BB%8B-%E5%AF%A6%E4%BD%9C-3-e8f1166cc09f)

## Result
![](https://github.com/Coolshanlan/Coronary-Artery-Segmentation/blob/master/demo_image/Result.png?raw=true)
## Medical Image Complete
[Grand Challenge](https://grand-challenge.org/)
## 自適應職方圖拓寬(色階調整)
https://www.itread01.com/content/1542310036.html

Automatic segmentation of coronary arteries using Gabor filters and thresholding based on multiobjective optimization

https://daneshyari.com/article/preview/557554.pdf

Gabor Image enhanced
## CT圖方向
![](https://github.com/Coolshanlan/Coronary-Artery-Segmentation/blob/master/demo_image/CT_Helper.png?raw=true)
## PyDicom 入門
[處理醫療影像的Python利器：PyDicom](https://zhuanlan.zhihu.com/p/59413289)
### RunTimeError raised StopIteration
Find the filereader file of dicom lib at your env and replace *raised StopIteration* to return

### Kaggle Complete
[Data Science Bowl 2017](https://www.kaggle.com/c/data-science-bowl-2017/overview)


## **名詞對照表**

**Computed Tomography** 電腦斷層

**Chambers** 腔室

**Atria** 心房

**Ventricle** 心室

**Artery** 血管/動脈

**Aorta** 主動脈/大動脈

**Ascending Aorta** 升 主動脈

**LM** (Left Main)

**LCA** (Left Coronary Aorta) 左冠狀動脈

**LAD** (Left Anterior Descending) 左前降支

**D1 D2** (Left Diagonal Artery) 左斜支

**LCx** (Left Circumflex) 左迴旋支

**OM** (Obtuse Marginal Branch) 邊緣支

**RCA** (Right Coronary Aorta) 右冠狀動脈

**PDA** (Posterior Descending Artery) 右後降支

**RMB** (Right Marginal Branch) 右邊緣支

**ramus intermedius** 中間支

![](https://www.researchgate.net/profile/George_Angelidis2/publication/322314679/figure/fig1/AS:640282919972865@1529666752438/Fig-1-Right-and-left-coronary-trees-LAD-left-anterior-descending-artery-LCx-left.png)
![](https://upload.wikimedia.org/wikipedia/commons/c/c9/Coronary_arteries.png)

## Tools
### ITK Tool
#### 3D segmentation
[Download](https://sourceforge.net/projects/itk-snap/files/itk-snap/3.6.0/itksnap-3.6.0-20170401-win64.exe/download)

Lower threshold 260 / Upper threshold 935
### 3D Slicer
[Download](https://www.slicer.org/)
#### Extension
https://www.slicer.org/wiki/Documentation/4.3/SlicerApplication/ExtensionsManager
#### Free tools
https://www.youtube.com/watch?v=ucnvE16pkmI

## 待看(可以看裡面enhance 的部分)
https://www.researchgate.net/publication/339900110_A_New_Strategy_to_Detect_Lung_Cancer_on_CT_Images
