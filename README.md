# Transformer Encoder
- BERT is powerful, and it was trained for nlp downstream tasks, but BERT is very large and not very fast to process the data.
- The benifits of building transformer encoder from scratch is that we can understand deeply how the mechanism works and customize BERT with fewer parameters.
- Code of transformer encoder is available [here](https://github.com/shuxg2017/transformer_encoder_demo/blob/master/encoders/transformer_encoder.py).
- [This notebook](https://github.com/shuxg2017/transformer_encoder_demo/blob/master/transformer_encoder_demo.ipynb) shows how to use the package. You can also modify the code for your own understanding.
- Note: there are 2 classes, the codes of **EncoderLayers** and **EncoderModel** are the same, **BUT**
   - EncoderLayers inherits from tf.keras.layers.Layer
   - EncoderModel inherits from tf.keras.Model
- I made the following graphs to show how to build your own encoding layer. You can follow along the [code](https://github.com/shuxg2017/tensorflow_transformer_encodingLayer/blob/master/encoders/transformer_encoder.py) with [these graphs](https://github.com/shuxg2017/tensorflow_transformer_encodingLayer/tree/master/multi_head_attention_example).<br>
**(Good Luck! :P)**
<hr>

### MultiHeadAttention Mechanism (MHA)

![MHA](https://github.com/shuxg2017/transformer_encoder_package/blob/master/multi_head_attention_example/mha.PNG)<br>
**I forgot to put attention weights in the tf.nn.softmax().**<br>
**"z" is the context matrix and it needs to be transposed and reshaped.**
![QKV](https://github.com/shuxg2017/transformer_encoder_package/blob/master/multi_head_attention_example/mha_qkv_op.PNG)

<hr>

### Encoding Layer

![Enc](https://github.com/shuxg2017/transformer_encoder_package/blob/master/multi_head_attention_example/encoder_layer.PNG)
