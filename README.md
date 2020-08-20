# Transformer Encoder
1. BERT is powerful, and it was trained for nlp downstream tasks, but BERT is very large and it is not very fast to process the data.
2. The benifits of building transformer encoder from scratch is that we can understand deeply how the mechanism works and customize BERT with fewer parameters.
3. Code of transformer encoder is available [here](https://github.com/shuxg2017/transformer_encoder_demo/blob/master/encoders/transformer_encoder.py).
4. To see how to use the package and visualize attention weights in [this notebook](https://github.com/shuxg2017/transformer_encoder_demo/blob/master/transformer_encoder_demo.ipynb). You can also modify the code for your own understanding.
6. Note: there are 2 classes, the codes of EncoderLayers and EncoderModel are the same, **BUT**
   - EncoderLayers inherits from tf.keras.layers.Layer
   - EncoderModel inherits from tf.keras.Model
7. I made the following graphs to show how to build your own encoding layer. You can follow along the code with these graphs.<br>
**(Good Luck! :P)**
<hr>

### MultiHeadAttention Mechanism (MHA)

![MHA](https://github.com/shuxg2017/transformer_encoder_package/blob/master/multi_head_attention_example/mha.PNG)
![QKV](https://github.com/shuxg2017/transformer_encoder_package/blob/master/multi_head_attention_example/mha_qkv_op.PNG)

<hr>

### Encoding Layer

![Enc](https://github.com/shuxg2017/transformer_encoder_package/blob/master/multi_head_attention_example/encoder_layer.PNG)
