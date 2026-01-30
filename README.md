# VIR_AJL_Team-Benzoyl-Peroxide

### **👥 Team Members**

| Name | GitHub Handle | Contribution |
| ----- | ----- | ----- |
| Channacy Un | @channacy | Trained different parameters for CNN, ResNet50, ConvNeXt
| Soraya Sardine | @Zrayaart | Trained CNN model and Assisted with ReadMe
| Joanne Liu | @joooanneliu | Trained ViT and EfficientNet_V2 models
| Nicole Rodriguez | @nicolerodriguez16 | 
| Jinglin Li| @jinglin-l | 
|Shreya Isharani| @shreyaisharani |


## **🎯 Project Highlights**

* Built a RegNet-Y model using PyTorch to classify 21 different skin conditions.
* Achieved a leaderboard F1-score of 0.69441, ranking top 4 in the Kaggle competition.
* Implemented a weighted loss function to address class imbalances.
* Applied fairness-aware data augmentation (contrast and brightness adjustments) to improve representation across diverse skin tones.

🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

How to clone the repository
1) Click on green code button in the upper right corner to copy repository url (preferably SSH)
2) Go to your command line / terminal and navigate using cd to your destination folder
3) Type git clone paste-your-repository-url (Note: may need to set up personal tokens for Github account, follow instructions on Github website)

How to access the dataset(s) and run notebook
1) Click on green code button and download zip file to see all train and test files
2) Upload the Jupyter Notebook (.ipynb file) to Google Colab
3) Create API token in Kaggle
a) Create and log in to your Kaggle Account
b) Click on your profile photo in the upper right corner and click on your profile
c) Click on settings in the upper right corner and scroll to API
d) Create new token and save in a secure place
4) Go back to Colab and click on the Secrets (Key icon) in the left sidebar
5) Create two secrets: KAGGLE_KEY, which will be the API token and KAGGLE_USERNAME, which is your Kaggle username. Toggle Notebook Access to allow for both rows
6) Now the notebook is ready to run. Go to Runtime in the top menubar and click “Run All”

---

## **🏗️ Project Overview**

* This project is part of the Break Through Tech AI Program, focused on equitable AI in dermatology.
* The challenge requires us to develop an AI model that can classify 21 different skin conditions from images while ensuring fair performance across different skin tones.
* Many dermatology AI models underperform on darker skin tones due to lack of diverse training data. This can lead to diagnostic disparities, misdiagnoses, and healthcare inequality. Our model aims to mitigate this issue by incorporating fairness techniques in training.

---

## **📊 Data Exploration**

Dataset Used
* Kaggle's competition dataset, containing 21 classes of skin conditions.
* Images labeled according to different dermatological diseases.


Data Preprocessing & Challenge
* Resized images to 224x224 pixels for model compatibility.
* Applied augmentations (random contrast, brightness adjustments, rotation, and flipping) to reduce bias.
* Balanced dataset using class weighting and a weighted sampler to prevent underrepresentation.
---

## **🧠 Model Development**

* Model Used
     * ConvNeXt
     * ViT
     * EfficientNet_V2
     * RegNet-Y 128GF (Pretrained on ImageNet, fine-tuned for skin condition classification).
     * Fully connected layer replaced with a classifier for 21 skin conditions.
* Weighted CrossEntropy Loss to handle class imbalances.
Feature Selection & Hyperparameter Tuning
* Tested different architectures (ResNet50, ConvNeXT, ViT, EfficientNet_V2)


- Best results achieved with ConvNeXT.
     * 0.72957 accuracy with 50 epochs, BATCH_SIZE = 256
     * 0.66841 accuracy with 50 epochs, BATCH_SIZE = 64
     * 0.68575 accuracy with BATCH_SIZE = 512
     * 0.69…With BATCH_SIZE = 128

* Fine-tuned last layers to improve performance.
Training Setup
* Training/Validation Split: 80% train, 20% validation.


* Loss Function: CrossEntropy Loss with class weights.


* Evaluation Metric: Weighted F1-score.

---

## **📈 Results & Key Findings**
Performance Metrics
* Leaderboard F1-Score: 0.72957.
Model Fairness Evaluation
* The model initially performed better on lighter skin tones.


* After bias mitigation, improvements were seen across darker skin tones.


* Used Explainability Tools (SHAP, Grad-CAM) to analyze misclassifications.

---

## **🖼️ Impact Narrative**
What steps did you take to address model fairness?
* Applied fairness-aware data augmentation (brightness, contrast, saturation).


* Used a weighted sampler to balance underrepresented skin tones.


* Evaluated fairness using AJL’s explainability tools.


What broader impact could your work have?
* Better AI-driven dermatology tools for underrepresented populations.


* Improved diagnostic accuracy across all skin tones.


* Potential for real-world clinical deployment to assist dermatologists.
---

## **🚀 Next Steps & Future Improvements**
Limitations of Current Model
* Still some performance discrepancies across skin tones.


* Dataset size constraints may limit generalization.


Future Improvements
* Use an ensemble of multiple architectures (ResNet, ConvNeXt).


* Explore adversarial debiasing techniques for better fairness.


* Incorporate larger, more diverse datasets for training.

---

## **📄 References & Additional Resources**

Kaggle Competition Page: [Equitable AI for Dermatology](https://www.kaggle.com/competitions/bttai-ajl-2025)

Algorithmic Justice League Guide: [AJL Fairness Guide](https://drive.google.com/file/d/1kYKaVNR_l7Abx2kebs3AdDi6TlPviC3q/view)

---


