# EHPAL-Net: Efficient Hybrid-fusion Physics-informed Attention Learning Network

Multimodal fusion learning offers a framework to jointly learn from heterogeneous data sources—a capability that is especially critical in healthcare, where AI models must integrate imaging, clinical records, and multi-omics to perform tasks like disease classification and survival prediction. Existing fusion strategies, however, suffer from three primary limitations:

1. **Ineffective Cross-Modal Interaction Modeling**  
   Most methods struggle to capture fine-grained, structure-preserving relationships across diverse modalities, leading to suboptimal shared representations.

2. **Limited Generalizability Across Modalities**  
   Many fusion architectures are tailored to specific data types (e.g., only imaging or only omics), hindering their ability to adapt when new modalities are introduced.

3. **High Computational Overhead**  
   Intermediate fusion layers often rely on large, high-dimensional attention matrices, which drastically increase parameter count and FLOPs—making deployment in resource-constrained environments impractical.

---

## Our Proposed Approach: EHPAL-Net

To overcome these challenges, we introduce the **Efficient Hybrid-fusion Physics-informed Attention Learning Network (EHPAL-Net)**—a lightweight, scalable framework designed for efficient, generalizable multimodal fusion in healthcare AI. Its core innovations include:

### 1. Efficient Hybrid Fusion (EHF) Layers

Each **EHF layer** processes one modality at a time in a sequential pipeline:

- **Inputs**:  
  - Current modality features: `x_i`  
  - Previous fused output: `x_{i-1}^S`

- **Steps within an EHF layer**:
  1. **Efficient Multimodal Residual Convolution (EMRC)**  
     - Captures multi-scale spatial cues from `x_i` and `x_{i-1}^S`.  
     - Uses lightweight residual blocks and heterogeneous convolutions to preserve fine-grained details without large parameter overhead.

  2. **Physics-informed Cross-modal Fusion Attention (PCMFA)**  
     - Models fine-grained hyperbolic and quantum-inspired interactions between modalities.  
     - Composed of:  
       - **Hyperbolic Quantum Mutual Guidance Attention (HQMGA)**  
         - Two parallel sub-blocks:  
           - **Poincaré Information Learning (PIL)**: Projects frequency-domain features into the hyperbolic Poincaré ball, preserving hierarchical structure.  
           - **Lorentz Information Learning (LIL)**: Projects frequency-domain features into the Lorentz hyperboloid, capturing complementary geometric relationships.  
       - **Multimodal Quantum-Inspired Attention (MQIA)**  
         - Maps modality-specific frequency components into a complex Hilbert space.  
         - Generates quantum-inspired attention weights that guide LIL via Minkowski inner-product modulation.  
       - **Multimodal Attention Fusion Gating (MAFG)**  
         - Learns to merge geometry-aware (hyperbolic) and quantum-guided attention weights.  
         - Produces a robust attention map `A_i` that refines cross-modal interactions.

  3. **Learnable Late Fusion (LF)**  
     - Integrates refined features from EMRC and PCMFA.  
     - Produces the fused output `x_i^S`, which is passed to the next EHF layer.

This **single-pass, non-iterative** design ensures both computational efficiency and robust representation learning.

---

### 2. Sequential Two-Phase Pipeline

EHPAL-Net operates in two main phases:

#### 2.1 Multimodal Shared Information Learning (MSIL)

- **Goal**: Fuse all `m` modalities into a single shared representation (`x_C`).  
- **Process**:
  1. **First EHF Layer**: Fuse modality `1` with modality `2` → `x_2^S`.  
  2. **Second EHF Layer**: Fuse `x_2^S` with modality `3` → `x_3^S`.  
  3. **Repeat** until all `m` modalities have been combined. The final shared representation is denoted as `x_C`.

#### 2.2 Heterogeneous Modality-Specific Multitask Learning (HMML)

- **Goal**: Use the final shared representation `x_C` to perform downstream tasks under a joint multitask objective.  
- **Components**:  
  - **Task Heads**:  
    - **Classification Head** (e.g., disease diagnosis)  
    - **Survival Prediction Head** (e.g., hazard regression)  
    - Any additional heads as needed (e.g., segmentation, regression, etc.)  
  - **Uncertainty Estimation**:  
    - Apply **Monte Carlo Dropout** at inference to quantify predictive uncertainty—crucial for high-risk medical decisions.  
  - **Joint Loss**:  
    - Combine classification loss (e.g., cross-entropy) and regression/survival loss (e.g., negative log-likelihood) into a single loss function, allowing end-to-end training of all heads simultaneously.

---

### 3. Physics-informed Cross-modal Fusion Attention (PCMFA)

The **PCMFA** module refines features produced by EMRC using two complementary attention mechanisms:

#### 3.1 Hyperbolic Quantum Mutual Guidance Attention (HQMGA)

- **Poincaré Information Learning (PIL)**  
  - Projects input features into a **hyperbolic Poincaré ball**.  
  - Captures hierarchical relationships by learning embeddings in negatively curved space.  
  - Frequency-domain information (via Discrete Cosine Transform + Global Average Pooling) is warped to the Poincaré manifold.  

- **Lorentz Information Learning (LIL)**  
  - Projects frequency-domain features into a **Lorentz hyperboloid**.  
  - Preserves complementary geometric cues (e.g., time-like vs. space-like separations).  
  - Learns Lorentzian embeddings via trainable curvature parameters.  

These two sub-blocks run **in parallel**, ensuring no information loss from cascading (as seen in purely cascaded architectures).

#### 3.2 Multimodal Quantum-Inspired Attention (MQIA)

- **Complex Hilbert Space Mapping**  
  - Transforms modality-specific frequency components into complex-valued embeddings.  
  - Embraces **quantum-inspired representations** to capture long-range dependencies.  

- **Quantum Mutual Guidance**  
  - Quantum attention weights are generated via inner-product rules in the complex Hilbert space.  
  - These weights guide the Lorentz Information Learning block, modulating the feature distributions via **Minkowski inner-product**.

#### 3.3 Multimodal Attention Fusion Gating (MAFG)

- **Inputs**:  
  - Hyperbolic attention map (from PIL/LIL)  
  - Quantum-inspired attention map (from MQIA)  
- **Learnable Gates**:  
  - Compute scalar gates that weigh geometry-aware vs. quantum-guided contributions.  
  - Element-wise fuse these two attention maps into a final attention map `A_i`.  
- **Output**:  
  - Refined attention map `A_i` applied to EMRC features, producing contextually enriched, cross-modal representations.

## 4. Addressing Key Challenges

To demonstrate how EHPAL-Net overcomes the three core limitations of existing multimodal fusion learning (MFL) methods, we describe our design choices in detail:

### 1. Effective Learning of Richer Complementary Shared Representations

- **Efficient Hybrid Fusion Strategy**  
  1. **EMRC (Efficient Multimodal Residual Convolution)**  
     - Captures multi-scale spatial details across each modality pair.  
  2. **PCMFA (Physics-informed Cross-modal Fusion Attention)**  
     - Focuses on learning fine-grained cross-modal interactions via an intermediate-fusion attention mechanism.  
  3. **Learnable Late Fusion (LF)**  
     - Integrates complementary cues from each modality’s EMRC+PCMFA outputs to preserve shared cross-modal patterns.  
  4. **SIR (Shared Information Refinement)**  
     - Enhances representational diversity by refining the fused shared features before passing to the next EHF layer.  

  > This cascaded design—EMRC ➔ PCMFA ➔ LF ➔ SIR—outperforms methods that rely solely on early, intermediate, late, or hybrid-early fusion, by capturing richer and more effective shared representations.

### 2. Efficient and Effective Design

- **Jointly Optimized Modules**  
  - **EMRC**, **PCMFA**, and **SIR** are architected to minimize parameter count and FLOPs while maximizing representational power:  
    - EMRC employs lightweight residual connections and heterogeneous convolutions for spatial feature extraction.  
    - PCMFA combines hyperbolic and quantum-inspired attention branches in parallel, avoiding large, iterative attention matrices.  
    - SIR uses a shallow refinement block to diversify shared features without significant computational overhead.  
- **Synergistic Performance–Computation Trade-off**  
  - By combining these three modules in a single-pass pipeline (no back-and-forth loops), EHPAL-Net achieves state-of-the-art accuracy with minimal latency—ideal for deployment in resource-constrained medical settings.

### 3. Generalization Across Heterogeneous Data Sources

- **Extensive Evaluation**  
  - EHPAL-Net is evaluated on **fifteen diverse medical datasets**, including:  
    - **Imaging**: HAM10000, SIPaKMeD, PathMNIST, OrganAMNIST, BraTS-2021, SARS-CoV-2 CT-Scan, CNMC-2019, Chest X-ray Pneumonia  
    - **Multi-omics**: TCGA-BRCA, TCGA-UCEC, TCGA-GBMLGG, TCGA-KIRP  
    - **EHR/Clinical**: MIMIC-III (MORT and ICD9 tasks), MHEALTH, UCI-HAR  
- **Scalability and Multitask Adaptability**  
  - The same EHF layers (EMRC ➔ PCMFA ➔ LF ➔ SIR) can fuse any combination of 2D imaging, tabular omics, and clinical time-series.  
  - Under the **HMML** phase, the final shared representation `x_C` feeds into multiple heads (e.g., disease classification, survival prediction) with Monte Carlo Dropout enabling reliable uncertainty estimation.

> By addressing each challenge—effective representation learning, efficient module design, and broad generalization—EHPAL-Net pushes the boundaries of multimodal fusion learning in healthcare AI.

---


<p><span style="color:blue;"><strong>NOTE:</strong></span></p>

<p><span style="color:blue;">For Multi-disease classification and mortality prediction, use this code:</span>  
<code>EHPAL-Net_for_multi-disease_classification_and_moratlity_prediction.ipynb</code></p>

<p><span style="color:blue;">For Multi-disease classification and patient survival and mortality predictions, use this code:</span>  
<code>EHPAL-Net_for_multi-disease_classification_and_patient_survival_and_moratlity_predictions.ipynb</code></p>



