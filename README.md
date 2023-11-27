# FL-Research-Highlight

**The purpose of this work is to follow the state-of-the-art works on Federated Learning**

[TOC]

## 1. Survey

- [Federated Learning for Generalization,Robustness, Fairness: A Survey and Benchmark](https://arxiv.org/pdf/2311.06750.pdf) - *ArXiv’23*

  They argued that **Generalization, Robustness, and Fairness** interact with each other to jointly enhance the practical federation deployment and this is the first work to simultaneously investigate the related research development and uniformly benchmark multi-view experimental analysis on the Generalization, Robustness, and Fairness realms.

- [Heterogeneous federated learning: State-of-the-art and research challenges](https://arxiv.org/pdf/2307.10616.pdf) - *ACM Computing Surveys‘23*

  They summarized the various research challenges in HFL from five aspects: **statistical heterogeneity, model heterogeneity, communication heterogeneity, device heterogeneity, and additional challenges.** They classify existing methods from three different levels according to the HFL procedure: **data-level, model-level, and server-level**. 



## 2. Baseline

- **FedAvg**: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) - *PMLR’17*

  Google built a basic paradigm for federated learning. 

  **Local SGD + Server average aggregation**



## 3. Model Regularization

- FedPox: [Federated optimization in heterogeneous networks](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html) - *MLSys’20*
- SCAFFOLD: [SCAFFOLD: Stochastic Controlled Averaging for On-Device Federated Learning](https://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf) - *ICML'20*
- MOON: [Model-Contrastive Federated Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.pdf) - *CVPR'21*



## 4. Optimization

- FedOPT: [Adaptive federated optimization](https://arxiv.org/pdf/2003.00295.pdf) - *ArXiv’21*
- MFL: [Accelerating Federated Learning via Momentum Gradient Descent](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9003425) - *IEEE TPDS'20*
- Mime: [Mime: Mimicking Centralized Stochastic Algorithms in Federated Learning](https://arxiv.org/pdf/2008.03606.pdf) - *ArXiv’21*
- Mimelite: [Breaking the centralized barrier for cross-device federated learning](https://proceedings.neurips.cc/paper_files/paper/2021/file/f0e6be4ce76ccfa73c5a540d992d0756-Paper.pdf) - *NeurIPS'21*
- FedGBO: [Accelerating Federated Learning With a Global Biased Optimiser](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9913718) - *IEEE TC'22*
- FedLocal: [Local Adaptivity in Federated Learning: Convergence and Consistency](https://arxiv.org/pdf/2106.02305.pdf) - *ArXiv’21*
- FedDA: [Accelerated Federated Learning with Decoupled Adaptive Optimization](https://proceedings.mlr.press/v162/jin22e/jin22e.pdf) - *PMLR'22*
- FedUR: [FedUR: Federated Learning Optimization Through Adaptive Centralized Learning Optimizers](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10180365) - *IEEE TSP'23*



## 5. Complex Aggregation Research：

### 5.1 Datasets

1.[Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data](https://arxiv.org/pdf/1811.11479.pdf) - *ArXiv’18*

They proposed **federated distillation (FD)** and **Federated Augmentation**, where each device collectively trains a **generative model (GAN)**, and thereby augments its local data towards yielding an IID dataset.



2.[FedHome: Cloud-Edge based Personalized Federated Learning for In-Home Health Monitoring](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9296274) - *IEEE TMC’20*

They proposed a novel cloud-edge based federated learning framework for in-home health monitoring, which learns a **shared global model** in the cloud from multiple homes at the network edges and achieves data privacy protection by keeping user data locally. They designed a **generative convolutional autoencoder (GCAE)** to achieve accurate and personalized health monitoring by refining the model with a generated class-balanced dataset from user's personal data.

...

### 5.2 Clients Selection

1.[Optimizing Federated Learning on Non-IID Data with Reinforcement Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9155494) - *IEEE INFORCOM’20* [code](https://github.com/iqua/flsim)

They proposed **FAVOR** an experience-driven control framework that **intelligently chooses the client devices** to participate in each round of federated learning to counterbalance the bias introduced by non-IID data and to speed up convergence. They propose a mechanism based on **deep Q-learning** that learns to select a subset of devices in each communication round to maximize a reward that encourages the increase of validation accuracy and penalizes the use of more communication rounds.



2.[Multi-Armed-Bandit-Based-Client-Scheduling-for-Federated-Learning](https://ieeexplore.ieee.org/abstract/document/9142401) - *IEEE TWC'20* **Cited by 188** [code](https://github.com/ramshi236/Multi-Armed-Bandit-Based-Client-Scheduling-for-Federated-Learning)

3.[Federated Learning with Class Imbalance Reduction](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9616052) - *EUSIPCO'21*

They designed an estimation scheme to reveal the **class distribution without the awareness of raw data**. They proposed a **multi-arm bandit** based algorithm that can select the client set with minimal class imbalance.



4.[TiFL: A Tier-based Federated Learning System](https://dl.acm.org/doi/pdf/10.1145/3369583.3392686) - *HPDC'20*

They proposed a Tier-based Federated Learning system, which **divides clients into tiers** based on their **training performance** and selects clients from the **same tier** in each training round to mitigate the straggler problem caused by heterogeneity in resource and data quantity. TiFL employs an **adaptive tier selection approach** to update the tiering on-the-fly based on the observed training performance and accuracy.



5.[FedSAE: A Novel Self-Adaptive Federated Learning Framework in Heterogeneous Systems](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9533876) - *IJCNN'21* [code](https://github.com/yasar-rehman/FedSAE)

They proposed FedSAE which leverages the **complete information of devices' historical training tasks** to predict the affordable training workloads for each device. In this way, FedSAE can estimate the reliability of each device and self-adaptively adjust the amount of training load per client in each round.



6.[HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients](https://arxiv.org/pdf/2010.01264.pdf) - *ICLR‘2021* [code](https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients)

They proposed an easy-to-implement framework HeteroFL that can **train heterogeneous local models** and aggregate them stably and effectively into a single global inference model.

...

### 5.3 Meta-learning

1.[Federated Meta-Learning with Fast Convergence and Efficient Communication](https://arxiv.org/abs/1802.07876) - *ArXiv’18* [code](https://github.com/ddayzzz/federated-meta)

They proposed a federated meta-learning framework FedMeta, where a parameterized algorithm (or meta-learner) is shared, instead of a global model in previous approaches.



2.[Personalized Federated Learning: A Meta-Learning Approach](https://arxiv.org/pdf/2002.07948.pdf) - *ArXiv’20* [code](https://github.com/ki-ljl/Per-FedAvg)

Per-FedAvg - [Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach](https://proceedings.neurips.cc/paper_files/paper/2020/file/24389bfe4fe2eba8bf9aa9203a44cdad-Paper.pdf) - *NeurIPS'20* [code](https://github.com/KarhouTam/Per-FedAvg)

They studied a personalized variant of the federated learning in which our goal is to find an initial shared model that current or new users can easily adapt to their local dataset by performing one or a few steps of gradient descent with respect to their own data.



3.pFedMe - [Personalized Federated Learning with Moreau Envelopes](https://proceedings.neurips.cc/paper_files/paper/2020/file/f4f1f13c8289ac1b1ee0ff176b56fc60-Paper.pdf) - *NeurIPS'20* [code](https://github.com/CharlieDinh/pFedMe)

They proposed an algorithm for personalized FL (pFedMe) using **Moreau envelopes** as clients' regularized loss functions, which help decouple personalized model optimization from the global model learning in a bi-level problem stylized for personalized FL.

...

### 5.4 Multi-task

1.MOCHA - [Federated Multi-Task Learning](https://proceedings.neurips.cc/paper_files/paper/2017/file/6211080fa89981f66b1a0c9d55c61d0f-Paper.pdf) - *NeurIPS'17* [code](https://github.com/gingsmith/fmtl)

They proposed a novel systems-aware optimization method MOCHA, which is robust to practical systems issues. Their method and theory for the first time consider issues of high communication cost, stragglers, and fault tolerance for distributed multi-task learning. 



2.[Three Approaches for Personalization with Applications to Federated Learning](https://arxiv.org/pdf/2002.10619.pdf) - *ArXiv'20* [code](https://github.com/sheetalreddy/Three-Approaches-for-Personalization-with-Applications-to-FederatedLearning)

They proposed and analyzed three approaches: **user clustering, data interpolation, and model interpolation**.



3.VIRTUAL - [Variational Federated Multi-Task Learning](https://arxiv.org/pdf/1906.06268.pdf) - *ArXiv’21* 

They introduced VIRTUAL, an algorithm for federated multi-task learning for general non-convex models.In VIRTUAL the federated network of the server and the clients is treated as a **star-shaped Bayesian network**, and learning is performed on the network using **approximated variational inference**.



4.FedAMP - [Personalized Cross-Silo Federated Learning on Non-IID Data](https://ojs.aaai.org/index.php/AAAI/article/view/16960) - *AAAI'21* 

They proposed FedAMP, a new method employing **federated attentive message passing** to facilitate similar clients to collaborate more. They established the convergence of FedAMP for both convex and non-convex models, and propose a **heuristic method** to further improve the performance of FedAMP when clients adopt deep neural networks as personalized models.



5.[Federated Multi-Task Learning under a Mixture of Distributions](https://proceedings.neurips.cc/paper_files/paper/2021/file/82599a4ec94aca066873c99b4c741ed8-Paper.pdf) - *NeurIPS'21* [code](https://github.com/omarfoq/FedEM)

They proposed to study federated MTL under the flexible assumption that each local data distribution is a mixture of unknown underlying distributions. This assumption encompasses most of the existing personalized FL approaches and leads to federated **EM-like algorithms** for both client-server and fully decentralized settings. Moreover, it provides a principled way to serve personalized models to clients not seen at training time.

...

### 5.5 Clustering

#### 5.5.1 Agglomerative

1.CFL - [Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9174890) - *IEEE TNNLS'20* [code](https://github.com/felisat/clustered-federated-learning)

They presented clustered FL (CFL), a novel federated multitask learning (FMTL) framework, which exploits **geometric properties of the FL loss surface** to group the client population into clusters with jointly trainable data distributions.



2.FL+HC - [Federated learning with hierarchical clustering of local updates to improve training on non-IID data](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9207469) - *IJCNN'20*

The presented a modification to FL by introducing a **hierarchical clustering step (FL+HC)** to separate clusters of clients by the **similarity of their local updates** to the global joint model.



#### 5.5.2 Kmean

**static clustering**

3.IFCA - [An Efficient Framework for Clustered Federated Learning](https://proceedings.neurips.cc/paper_files/paper/2020/file/e32cc80bf07915058ce90722ee17bb71-Paper.pdf) *NeurIPS'20* [code](https://github.com/jichan3751/ifca)

They proposed a new framework dubbed the Iterative Federated Clustering Algorithm (IFCA), which **alternately estimates the cluster identities** of the users and optimizes model parameters for the user clusters via **gradient descent**.



4.FeSEM - [Multi-center federated learning: clients clustering for better personalization](https://link.springer.com/article/10.1007/s11280-022-01046-x) - *World Wide Web'23* [code](https://github.com/mingxuts/multi-center-fed-learning)

They proposed a novel **multi-center aggregation mechanism** to cluster clients using their models' parameters. It learns multiple global models from data as the cluster centers, and simultaneously derives the optimal matching between users and centers. They then formulate it as an optimization problem that can be efficiently solved by a stochastic expectation maximization (EM) algorithm.



5.FlexCFL - [Flexible Clustered Federated Learning for Client-Level Data Distribution Shift](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9647969) - *TPDS'21* [code](https://github.com/morningD/FlexCFL)

FlexCFL leverages a novel decomposed data-driven measure called **euclidean distance of Decomposed Cosine similarity (EDC)** for client clustering. Another design that makes FlexCFL more practical is we **maintain an auxiliary server** to address the cold start issue of new devices. Furthermore, FlexCFL can detect the client-level data distribution shift based on **Wasserstein distance** and migrate clients with affordable communication.



**semi-dynamic clustering**

6.FedSim - [FedSim: Similarity guided model aggregation for Federated Learning](https://www.sciencedirect.com/science/article/abs/pii/S0925231221016039) - *Neurocomputing'22* [code](https://github.com/chamathpali/FedSim)

FedSim decomposes FL aggregation into local and global steps. Clients with **similar gradients** are clustered to provide local aggregations, which thereafter can be globally aggregated to ensure better coverage whilst reducing variance.

...



### 5.6 Neurons Matching

1.FedMA - [Federated Learning with Matched Averaging](https://arxiv.org/pdf/2002.06440.pdf) - *ArXiv'20* [code](https://github.com/IBM/FedMA)

FedMA constructs the **shared global model** in a layer-wise manner by **matching and averaging hidden elements** (i.e. channels for convolution layers; hidden states for LSTM; neurons for fully connected layers) with similar feature extraction signatures.



2.FedDist - [A Federated Learning Aggregation Algorithm for Pervasive Computing: Evaluation and Comparison](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9439129) - *PerCom'21* [code](https://github.com/getalp/PerCom2021-FL)

They propose a novel aggregation algorithm, termed FedDist, which is able to modify its model architecture (here, deep neural network) by **identifying dissimilarities between specific neurons** amongst the clients. This permits to account for clients' specificity without impairing generalization. 



### 5.7 Hypernetworks

1.pFedLA - [Layer-wised Model Aggregation for Personalized Federated Learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_Layer-Wised_Model_Aggregation_for_Personalized_Federated_Learning_CVPR_2022_paper.pdf) - *CVPR'21* [code](https://github.com/KarhouTam/pFedLA)

They proposed a **dedicated hypernetwork** per client on the server side, which is trained to identify the mutual contribution factors at layer granularity. Meanwhile, a **parameterized mechanism** is introduced to update the layer-wised aggregation weights to progressively exploit the inter-user similarity and realize accurate model personalization.

2.pFedHN - [Personalized Federated Learning using Hypernetworks](https://proceedings.mlr.press/v139/shamsian21a/shamsian21a.pdf) - *PMLR'21* [code](https://github.com/AvivSham/pFedHN)

In this approach, **a central hypernetwork model** is trained to generate a set of models, one model for each client. This architecture provides effective parameter sharing across clients while maintaining the capacity to generate unique and diverse personal models.

...



### 5.8 Fairness and robustness

1.Ditto - [Ditto: Fair and Robust Federated Learning Through Personalization](https://proceedings.mlr.press/v139/li21h/li21h.pdf) - *PMLR’21* [code](https://github.com/s-huu/Ditto)

They propose employing a simple, general framework for personalized federated learning, Ditto, that can inherently provide **fairness and robustness** benefits, and develop a scalable solver for it.



2.FAIR - [FAIR: Quality-Aware Federated Learning with Precise User Incentive and Model Aggregation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9488743) - *IEEE INFOCOM'21*

They proposed a novel system named FAIR. FAIR integrates three major components: 1) learning quality estimation: we leverage **historical learning records** to estimate the user learning quality; 2) quality-aware incentive mechanism: within the recruiting budget, they model a **reverse auction problem** to encourage the participation of high-quality learning users; and 3) model aggregation: we devise an aggregation algorithm that integrates the **model quality into aggregation** and filters out non-ideal model updates, to further optimize the global learning model.



3.FedPAC - [Personalized federated learning with feature alignment and classifier collaboration](https://arxiv.org/pdf/2306.11867.pdf) - *ICLR'23* [code](https://github.com/JianXu95/FedPAC)

They conducted explicit **local-global feature alignment** by leveraging **global semantic knowledge** for learning a better representation. Moreover, we quantify the benefit of classifier combination for each client as a function of the combining **weights and derive an optimization problem** for estimating optimal weights.

...



### 5.9 Others

...



## 6. Datasets

[LEAF: A Benchmark for Federated Settings](https://arxiv.org/pdf/1812.01097.pdf) - *ArXiv'19*

### MNIST

A 10-class handwritten digits image classification task, which is divided into 1,000 clients, each with only two classes of digits.

### FEMNIST

A handwritten digits and characters image classification task, which is built by resampling the EMNIST according to the writer and downsampling to 10 classes ('a'-'j').

### Synthetic

It's a synthetic federated dataset.

### FashionMNIST

A 28*28 grayscale images classification task, which comprises 70,000 fashion products from 10 categories.

### Sentiment140

A tweets sentiment analysis task, which contains 772 clients, each client is a different Twitter account.













