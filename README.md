# Text De-toxification

## Personal info
Polina Lesak, p.lesak@innopolis.university, BS21-DS-01(exchange student)

## Solution
The repository contains code for detoxifying text by ~80%.
In the course of solving the problem, 3 hypotheses were put forward and implemented. A detailed report on the algorithms can be found in the ./reports/Final_report.pdf

1 Hypothesis: create simple vocabulary(vocab.json) and use it on test dataset to detoxify it.

2 Hypothesis: use BERT model and "smart" vocabulary on test dataset to detoxify it.

Final solution: Take the least toxic option is selected from the two predicted data sets and saved to final solution dataset.

![Compare toxicity level of difference algorithms](https://github.com/polinaLesak/text-detoxification/blob/master/reports/figures/compare_toxicity.png)

tox_result - toxicity level of final solution translation text

tox_vocab - toxicity level of vocab-translated text

ref_tox - toxicity level of BERT-translated text

ref_tox - toxicity level of reference text
## Main processes


### Transforming data

To download and prepare test and train datasets use this code:

`python ./src/data/make_dataset.py`


### Making predictions

To make predictions using simple vocab:

`python ./src/models/vocab_predict.py`

To make prediction using "smart" vocab and BERT model:

`python ./src/models/condbert_predict.py`





