# TF-Speech-Recognition
Experimental code for training CLDNN models using tensorflow.

### **Model Architecture**
- 2 Conv layers with larger filter size + 5 residual block (15 conv layers) + 5 residual LSTM 

### Results on LibriSpeech
- Trained on LibriSpeech-train-clean-100

|Validation-Set                | WER       |Link       |
|-----------------|:--------:|:--------:|
|Librispeech-test-clean| NA    | NA|

- Training logs
![](https://github.com/ZhishengWang/TF-Speech-Recognition/edit/master/img/loss.png)
