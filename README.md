# HEDGE
Code for the paper ["Generating Hierarchical Explanations on Text Classification via Feature Interaction Detection"](https://arxiv.org/abs/2004.02015)

### Requirement:
- torchtext == 0.4.0
- gensim == 3.4.0
- pytorch == 1.2.0
- numpy == 1.16.4

### Model and data:
Download well-trained [models and data](https://drive.google.com/drive/folders/1_ME4CbVsDGt_UBqwu8Df7m9CsAut5IXZ?usp=sharing).

### Generate explanations:

We provide the example code of HEDGE interpreting the LSTM, CNN and BERT model on the IMDB dataset. We adopt the BERT-base model built by huggingface: https://github.com/huggingface/transformers.

In each folder, run the following command to generate explanations on the test data for a well-trained model.
```
python hedge_main_model_imdb.py --save /path/to/your/model
```
We save the start-end word indexes of text spans in a hierarchy into the "hedge_interpretation_index.txt" file.

To visualize the hierarchical explanation of a sentence, run
```
python hedge_main_model_imdb.py --save /path/to/your/model --visualize 1(the index of the sentence)
```

### Reference:
If you find this repository helpful, please cite our paper:
```bibtex
@inproceedings{chen2020generating,
  title={Generating hierarchical explanations on text classification via feature interaction detection},
  author={Chen, Hanjie and Zheng, Guangtao and Ji, Yangfeng},
  booktitle={ACL},
  url={https://arxiv.org/abs/2004.02015},
  year={2020}
}
```
