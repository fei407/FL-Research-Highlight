# FL-Research-Highlight

**The purpose of this work is to follow the state-of-the-art works on Federated Learning**

## Survey

- [Federated Learning for Generalization,Robustness, Fairness: A Survey and Benchmark](https://arxiv.org/pdf/2311.06750.pdf) - *ArXiv’23*

  They argue that **Generalization, Robustness, and Fairness** interact with each other to jointly enhance the practical federation deployment and this is the first work to simultaneously investigate the related research development and uniformly benchmark multi-view experimental analysis on the Generalization, Robustness, and Fairness realms.

- [Heterogeneous federated learning: State-of-the-art and research challenges](https://arxiv.org/pdf/2307.10616.pdf) - *ACM Computing Surveys‘23*

  They summarize the various research challenges in HFL from five aspects: **statistical heterogeneity, model heterogeneity, communication heterogeneity, device heterogeneity, and additional challenges.** They classify existing methods from three different levels according to the HFL procedure: **data-level, model-level, and server-level**. 

## Baseline

- **FedAvg**: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) - *PMLR’17*

  Google built a basic paradigm for federated learning. 

  **Local SGD + Server average aggregation**

#### Model regularization

- FedPox: [Federated optimization in heterogeneous networks](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html) - *MLSys’20*
- SCAFFOLD: [SCAFFOLD: Stochastic Controlled Averaging for On-Device Federated Learning](https://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf) - *ICML'20*
- MOON: [Model-Contrastive Federated Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.pdf) - *CVPR'21*



## Complex Aggregation Research：

**Datasets** 每个客户端选一部分数据上传，训练GAN/GCAE模型，根据GAN/GCAE生成增强的IID数据集

GAN - [Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data](https://arxiv.org/pdf/1811.11479.pdf) - *ArXiv’23*

Generative Convolutional AutoEncoder (GCAE) model - [FedHome: Cloud-Edge based Personalized Federated Learning for In-Home Health Monitoring](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9296274) - *IEEE TMC’20*

...

**Clients Selection** 选择具有同样数据分布的客户端加速收敛

Deep Q-learning - [Optimizing Federated Learning on Non-IID Data with Reinforcement Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9155494) - *IEEE INFORCOM’20*

MAB - [Federated Learning with Class Imbalance Reduction](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9616052) - *EUSIPCO'21*

Adaptive Selection [TiFL: A Tier-based Federated Learning System](https://dl.acm.org/doi/pdf/10.1145/3369583.3392686) - *HPDC'20*

[FedSAE: A Novel Self-Adaptive Federated Learning Framework in Heterogeneous Systems](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9533876) - *IJCNN'21*

[HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients](https://arxiv.org/pdf/2010.01264.pdf) - *ArXiv’21*

...

**Meta-learning** 元学习

[Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach](https://proceedings.neurips.cc/paper_files/paper/2020/file/24389bfe4fe2eba8bf9aa9203a44cdad-Paper.pdf) - *NeurIPS'20*

[Personalized Federated Learning with Moreau Envelopes](https://proceedings.neurips.cc/paper_files/paper/2020/file/f4f1f13c8289ac1b1ee0ff176b56fc60-Paper.pdf) - *NeurIPS'20*

[Multi-Layer Personalized Federated Learning for Mitigating Biases in Student Predictive Analytics](https://arxiv.org/pdf/2212.02985.pdf) - *ArXiv’22*

...

**Multi-task** 多任务

[Federated Multi-Task Learning](https://proceedings.neurips.cc/paper_files/paper/2017/file/6211080fa89981f66b1a0c9d55c61d0f-Paper.pdf) - *NeurIPS'17*

[Variational Federated Multi-Task Learning](https://arxiv.org/pdf/1906.06268.pdf) - *ArXiv’21*

[Personalized Cross-Silo Federated Learning on Non-IID Data](https://ojs.aaai.org/index.php/AAAI/article/view/16960) - *AAAI'21*

[Federated Multi-Task Learning under a Mixture of Distributions](https://proceedings.neurips.cc/paper_files/paper/2021/file/82599a4ec94aca066873c99b4c741ed8-Paper.pdf) - *NeurIPS'21*

[Ditto: Fair and Robust Federated Learning Through Personalization](https://proceedings.mlr.press/v139/li21h/li21h.pdf) - *PMLR’21*

[Three Approaches for Personalization with Applications to Federated Learning](https://arxiv.org/pdf/2002.10619.pdf) - *ArXiv'20*

...

**Clustering** 聚类方法

[Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9174890) - *IEEE TNNLS'20*

[Federated learning with hierarchical clustering of local updates to improve training on non-IID data](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9207469) - *IJCNN'20*

[An Efficient Framework for Clustered Federated Learning](https://proceedings.neurips.cc/paper_files/paper/2020/file/e32cc80bf07915058ce90722ee17bb71-Paper.pdf) *NeurIPS'20*

[Patient Clustering Improves Efficiency of Federated Machine Learning to predict mortality and hospital stay time using distributed Electronic Medical Records](https://www.sciencedirect.com/science/article/pii/S1532046419302102) - *Journal of Biomedical Informatics'19*

[FedGroup: Efficient Federated Learning via Decomposed Similarity-Based Clustering](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9644782) - *BdCloud'21*

[Multi-Center Federated Learning](https://link.springer.com/article/10.1007/s11280-022-01046-x) - *World Wide Web'23*

[FedSim: Similarity guided model aggregation for Federated Learning](https://www.sciencedirect.com/science/article/abs/pii/S0925231221016039) - *Neurocomputing'22*

[Clustered federated learning with weighted model aggregation for imbalanced data](Clustered federated learning with weighted model aggregation for imbalanced data) - *China Communication'22*

[Adaptive Clustering-Based Model Aggregation for Federated Learning with Imbalanced Data](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9593144) - *SPAWC'21*

[Federated Learning Model Training Method Based on Data Features Perception Aggregation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9625291) - *IEEE VTC'21*

...

**Sharing** 共享一部分模型

[Exploiting Shared Representations for Personalized Federated Learning](https://proceedings.mlr.press/v139/collins21a/collins21a.pdf) - *PMLR’21*

[Personalized federated learning with feature alignment and classifier collaboration](https://arxiv.org/pdf/2306.11867.pdf) - *ArXiv'23*

...

**Hypernetworks** 超网络

[Layer-wised Model Aggregation for Personalized Federated Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_Layer-Wised_Model_Aggregation_for_Personalized_Federated_Learning_CVPR_2022_paper.pdf) - *CVPR'21*

[Personalized Federated Learning using Hypernetworks](https://proceedings.mlr.press/v139/shamsian21a/shamsian21a.pdf) - *PMLR'21*

...

**Others** 其他

[Federated Learning with Matched Averaging](https://arxiv.org/pdf/2002.06440.pdf) - *ArXiv'20*

[A Federated Learning Aggregation Algorithm for Pervasive Computing: Evaluation and Comparison](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9439129) - *PerCom'21*

[Federated Learning with Personalization Layers](https://arxiv.org/pdf/1912.00818.pdf) - *ArXiv'19*

[FAIR: Quality-Aware Federated Learning with Precise User Incentive and Model Aggregation]() - *IEEE INFOCOM'21*

...



## Datasets

[LEAF: A Benchmark for Federated Settings](https://arxiv.org/pdf/1812.01097.pdf) - *ArXiv'19*













