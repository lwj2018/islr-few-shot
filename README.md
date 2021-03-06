# Experiment result
## Standard few-shot learning on Isolated Chinese Sign Language Dataset
- Train setting: 20-way 5-shot
- The experiment result of PN, RN, MN is base on our implementation

| Model                       | 5-way 1-shot | 5-way 5-shot | 10-way 1-shot | 10-way 5-shot |
| --------------------------- | ------------ | ------------ | ------------- | ------------- |
| PN                          |    82.69     |    95.11     |    70.36      |    91.45      |
| RN                          |    91.37     |    96.85     |    85.01      |    93.90      |
| MN                          |    85.76     |    95.79     |    79.78      |    93.10      |
| GCR                         |    79.66     |    83.66     |    70.66      |    78.64      |
| GCR+relation                |    90.11     |    92.87     |    84.23      |    90.27      |
| GCR+relation+induction      |    95.78     |    97.63     |    91.76      |    95.67      |
## General few-shot learning on Isolated Chinese Sign Language Dataset
- Test with 5-way 5-shot model
- Accu_a is classify accuracy on all categories
- Accu_b is classify accuracy on base categories
- Accu_n is classify accuracy on novel categories

| Model       | Accu_a | Accu_b | Accu_n |
| ----------- | ------ | ------ | ------ |
| PN          | 45.60  | 54.88  |  8.50  |
| RN          | 53.40  | 58.48  | 33.10  |
| MN          | 73.84  | 83.65  | 34.60  |
| Ours        | 74.14  | 79.90  | 51.10  |
## Influence of the size of training set
- Training for 10 epochs

| Samples reserved     | Accu_a | Accu_b | Accu_n |
| -------------------- | ------ | ------ | ------ |
| 10                   | 61.46  | 64.28  | 50.20  |
| 20                   | 66.56  | 70.95  | 49.00  |
| 30                   | 70.16  | 74.75  | 51.80  |
| 40                   | 74.14  | 79.90  | 51.10  |
## Influence of the embedding dim
- Training for 10 epochs

| embedding dim        | Accu_a | Accu_b | Accu_n |
| -------------------- | ------ | ------ | ------ |
| 256                  | 73.44  | 78.00  | 55.20  |
| 192                  | 72.68  | 76.63  | 56.90  |
| 128                  | 73.40  | 77.83  | 55.70  |
| 96                   | 70.16  | 74.55  | 52.60  |
| 64                   | 66.64  | 71.23  | 48.30  |
| 32                   | 61.22  | 65.30  | 44.90  |

Accuracies of pretrained HCN

| embedding dim        | Accu_a | Accu_b | Accu_n |
| -------------------- | ------ | ------ | ------ |
| 256                  | 80.68  | 94.23  | 26.50  |
| 192                  | 76.72  | 87.23  | 34.70  |
| 128                  | 76.12  | 87.03  | 32.50  |
| 96                   | 74.66  | 85.90  | 29.70  |
| 64                   | 73.20  | 85.65  | 23.40  |
| 32                   | 68.38  | 82.60  | 11.50  |

## New Ablation Study
- train_epochs = 20
### Standard FSL
| Model                       | 5-way 1-shot | 5-way 5-shot | 10-way 1-shot | 10-way 5-shot |
| --------------------------- | ------------ | ------------ | ------------- | ------------- |
| Without L_{gfsl}            |    87.70     |    97.15     |    79.45      |    94.40      |
| GCR                         |    87.08     |    98.05     |    79.14      |    95.24      |
| GCR+relation                |    90.32     |    98.23     |    83.78      |    96.26      |
| GCR+relation+induction      |    91.97     |    98.37     |    85.96      |    96.52      |
### Generalized FSL
| Model                       | Accu_a | Accu_b | Accu_n |
| --------------------------- | ------ | ------ | ------ |
| Without L_{gfsl}            |  71.66 |  80.63 |  35.80 |
| GCR                         |  73.08 |  81.03 |  41.30 |
| GCR+relation                |  70.90 |  76.45 |  48.70 |
| GCR+relation+induction      |  73.56 |  79.08 |  51.50 |

## Tune model for C_{base}
| Model                       | Accu_a | Accu_b | Accu_n |
| --------------------------- | ------ | ------ | ------ |
| r1 800, r2 256              |  73.60 |  79.23 |  51.10 |
| r1 1600, r2 512             |  72.26 |  77.93 |  49.60 |