# EHPAL-Net: Efficient Hybrid-fusion Physics-informed Attention Learning Network

Multimodal fusion learning offers a framework to jointly learn from heterogeneous data sources—a capability that is especially critical in healthcare, where AI models must integrate imaging, clinical records, and multi-omics to perform tasks like disease classification and survival prediction. Existing fusion strategies, however, suffer from four primary limitations:

1. **Limited Robust Representations Learning**  
   Most methods have limitations to capture fine-grained, structure-preserving relationships across diverse modalities, leading to suboptimal shared representation learning.

2. **Limited Generalizability Across Modalities**  
   Many fusion architectures are tailored to specific data types (e.g., only imaging or only omics), hindering their ability to adapt when new modalities are introduced.

3. **High Computational Overhead**  
   Intermediate fusion layers often rely on large, high-dimensional attention matrices, which drastically increase parameter count and FLOPs—making deployment in resource-constrained environments impractical.

4. **Assumption of Sample-wise Alignment**  
   A common shortcoming of many multimodal fusion approaches is the assumption that each modality is drawn from the same patient cohort. In contrast, EHPAL-Net is trained and evaluated on a suite of completely unpaired, heterogeneous datasets (e.g., HAM10000, SIPaKMeD, CNMC, BraTS-2021, KVASIR, MIT-BIH, TCGA-BRCA, KIRP, UCEC), each collected under different protocols and patient populations. This heterogeneity introduces two primary challenges:  
   - **Absence of Sample-wise Correspondence**: Without paired examples, there is no direct supervision for mapping features in one modality to those in another.  
   - **Distributional Shifts Across Cohorts**: Varying demographic, imaging, and noise characteristics across datasets can undermine naive fusion strategies.  


---

## Our Proposed Approach: EHPAL-Net

To overcome these challenges, we introduce the **Efficient Hybrid-fusion Physics-informed Attention Learning Network (EHPAL-Net)**—a lightweight, scalable framework designed for efficient, generalizable multimodal fusion in healthcare AI. 

## Addressing Key Challenges by EHPAL-Net:

### 1. Effective Learning of Richer Robust Shared Representations

- **Efficient Hybrid Fusion Strategy**  
  1. **EMRC (Efficient Multimodal Residual Convolution)**  
     - Captures multi-scale spatial details across each modality pair.  
  2. **PCMFA (Physics-informed Cross-modal Fusion Attention)**  
     - Focuses on learning fine-grained cross-modal interactions via an intermediate-fusion attention mechanism.  
  3. **SIR (Shared Information Refinement)**  
     - Enhances representational diversity by refining the fused shared features before passing to the next EHF layer.  

  > This cascaded design—EMRC ➔ PCMFA ➔ SIR—outperforms methods that rely solely on early, intermediate, late, or hybrid-early fusion, by capturing richer and more effective shared representations.

### 2. Efficient and Effective Design

- **Jointly Optimized Modules: EMRC, PCMFA, and SIR to balance Performance–Computation Trade-off for resource-constrained healthcare environments**  
  
### 3. Generalization Across Heterogeneous Data Sources

- **Extensive Evaluation**  
  - EHPAL-Net is evaluated on **fifteen diverse medical datasets**, including:  
    - **Imaging**: HAM10000, SIPaKMeD, PathMNIST, OrganAMNIST, BraTS-2021, SARS-CoV-2 CT-Scan, CNMC-2019, Chest X-ray Pneumonia  
    - **Multi-omics**: TCGA-BRCA, TCGA-UCEC, TCGA-GBMLGG, TCGA-KIRP  
    - **EHR/Clinical**: MIMIC-III (MORT and ICD9 tasks), MHEALTH, UCI-HAR  

> By addressing each challenge—effective representation learning, efficient module design, and broad generalization—EHPAL-Net pushes the boundaries of multimodal fusion learning in healthcare AI.

### 4. Datasets and Links

- [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [Computer Science Department, University of Ioannina](https://www.cs.uoi.gr/)
- [MedMNIST](https://medmnist.com/)
- [MedMNIST (Documentation)](https://medmnist.com/)
- [RSNA-ASNR-MICCAI-BRATS 2021 Dataset](https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/)
- [SARS-CoV-2 CT Scan Dataset](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset)
- [C-NMC 2019 Collection](https://www.cancerimagingarchive.net/collection/c-nmc-2019/)
- [Chest X-ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [TCGA-BRCA Collection](https://www.cancerimagingarchive.net/collection/tcga-brca/)
- [TCGA-UCEC Collection](https://www.cancerimagingarchive.net/collection/tcga-ucec/)
- [TCGA-GBM Collection](https://www.cancerimagingarchive.net/collection/tcga-gbm/)
- [TCGA-KIRP Collection](https://www.cancerimagingarchive.net/collection/tcga-kirp/)
- [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/)
- [UCI MHEALTH Dataset](https://archive.ics.uci.edu/dataset/319/mhealth+dataset)
- [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones/)

---


<p><span style="color:blue;"><strong>NOTE:</strong></span></p>

<p><span style="color:blue;">For Multi-disease classification and mortality prediction, use this code:</span>  
<code>EHPAL-Net_for_multi-disease_classification_and_moratlity_prediction.ipynb</code></p>

<p><span style="color:blue;">For Multi-disease classification and patient survival and mortality predictions, use this code:</span>  
<code>EHPAL-Net_for_multi-disease_classification_and_patient_survival_and_moratlity_predictions.ipynb</code></p>


