# FL-Research-Highlight

**The purpose of this work is to follow the state-of-the-art works on Federated Learning. **

## Survey

- [Federated Learning for Generalization,Robustness, Fairness: A Survey and Benchmark](https://arxiv.org/pdf/2311.06750.pdf) - *arXiv:2311.06750’23*

  Authors argue that **Generalization, Robustness, and Fairness** interact with each other to jointly enhance the practical federation deployment and this is the first work to simultaneously investigate the related research development and uniformly benchmark multi-view experimental analysis on the Generalization, Robustness, and Fairness realms.

- [Heterogeneous federated learning: State-of-the-art and research challenges](https://arxiv.org/pdf/2307.10616.pdf) - *ACM Computing Surveys‘23*

## Baseline

- **FedAvg**: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) - *PMLR’17*

  Google built a basic paradigm for federated learning. 

  **Local SGD + Server average aggregation**

## Specific Works

### 1 联邦优化

#### 正则化

- FedPox: [Federated optimization in heterogeneous networks](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html) - *MLSys’20*
- SCAFFOLD: [SCAFFOLD: Stochastic Controlled Averaging for On-Device Federated Learning](https://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf) - *ICML20*
- MOON: [Model-Contrastive Federated Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.pdf) - *CVPR'21*

#### 元学习



#### 多任务



### 2 Server-Level Methods

#### 2.1 Client Selection

uniform data distributions, network bandwidth, computation capability, local resources
**考虑数据分布**

- alleviate the bias introduced by Non-IID

- Favor: deep reinforcement learning

- class imbalance problem:  a client selection algorithm for the minimal class imbalance

- correlation-based client selection strategy: Gaussian process

- synergy协同性

**处理硬件和网络**

- FedSAE: client training history tasks, selects the clients with higher values
- FedCS: differences in data resources, computing capabilities, and wireless channel conditions
- TiFL: adaptive layer selection method
- multi-layer online coordination framework
- HybridFL: specific probability distribution depending on the region slack factor

#### 2.1 Client Clustering

- a hierarchical clustering step
- multi-center aggregation mechanism
- FeSEM: employs Stochastic Expectation Maximization (SEM)
- FedFMC: dynamically groups devices of similar prototypes in certain epochs
- CFL: cosine similarity between their gradient updates
- Iterative Federated Clustering Algorithm (IFCA) 



## Datasets

### 图像分类

#### Cifar-10

#### Cifar-100

#### Tiny-ImageNet

#### Fashion-MNIST



### 词汇预测

#### 莎士比亚









