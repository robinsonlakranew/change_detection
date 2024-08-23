# Remote Sensing Change Detection

#### **Project Goal:**
This project is the initial step towards a broader goal of understanding the effects of deforestation and wildfires. By focusing on wildfire detection through change detection, this project aims to lay the groundwork for future research and development.

#### **Problem Statement:**
Detecting changes in land cover, for satellite imagery, using pyramid spatial–temporal attention module.

#### **Dataset:**
Due to the unavailability of open-source labelled datasets specifically tailored for wildfire detection and its long-term impacts, this project initially utilizes a publicly available change detection dataset to establish a baseline. The dataset provides labeled images that can be used to train and evaluate change detection models. 

In subsequent phases, a custom dataset will be collected to address the specific requirements of deforestation/wildfire detection and its environmental consequences.

#### **Methodology:**

**1. Dataset used for training: LEVIR-CD**, it consists of a large set of bitemporal Google Earth images, with 637 image pairs (1024 × 1024) and over 31 k independently labeled change instances

**2. Model Structure**

The model consists of two primary components:

* A Siamese FCN for extracting bitemporal image feature maps.
* A self-attention module for refining these feature maps by considering spatial-temporal dependencies.

**Feature Extractor**

* Based on the ResNet-18 architecture.
* Global pooling and fully connected layers are removed to adapt for dense classification tasks. High-level and low-level features are fused for better representations.

**Self-Attention Mechanism**

* Calculates correlations between different elements within the image.
* Exploits spatial-temporal dependencies to improve illumination invariance and 		misregistration robustness.
* Inspired by the PSPNet approach for incorporating global spatial context.
* Uses multi-scale subregions to capture dependencies at various scales.
* Reduces the impact of misregistration errors by considering global relationships between objects.

3. Employed a **contrastive loss** to encourage a small distance of each no-change pixel pair and a large distance for each change in the embedding space.

4. **Addressing Class Imbalance with Batch-Balanced Contrastive Loss**: BCL addresses this by using a batch-weight prior to modify the class weights in the original contrastive loss. This helps to balance the influence of different classes and mitigate the impact of class imbalance.


#### **Open [change_detection.ipqnb](https://github.com/robinsonlakranew/change_detection/blob/main/change_detection.ipynb) to see model implementation and results**


#### **Trained Models:**
Links for trained model checkpoints:
1. Feature Extractor :https://drive.google.com/uc?export=download&id=1QOjoa-YFvHOgCLXaxvZUlrlidihpmbWU
2. Attention Module: https://drive.google.com/uc?export=download&id=196bR8LdSfVawnSOVmM_YPW25UetucUYX

#### **Future Work:**
The knowledge gained from this project will be applied to create a custom dataset focused on wildfire detection and its environmental implications. The developed model will be refined and expanded to include additional factors such as vegetation indices, weather data, and fire spread modeling.
