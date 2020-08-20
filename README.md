# Transformer Encoder
1. Build [transformer encoder package](https://github.com/shuxg2017/transformer_encoder_demo/blob/master/encoders/transformer_encoder.py).
2. Show how to use the package and visualize attention weights in [this notebook](https://github.com/shuxg2017/transformer_encoder_demo/blob/master/transformer_encoder_demo.ipynb).
3. Goal is to customize BERT, and original BERT is too large.
4. You can modify [this package](https://github.com/shuxg2017/transformer_encoder_demo/blob/master/encoders/transformer_encoder.py).
5. Note: there are 2 classes, the codes of EncoderLayers and EncoderModel are the same, **BUT**
   - EncoderLayers inherits from tf.keras.layers.Layer
   - EncoderModel inherits from tf.keras.Model
6. I made the following graphs to show how to build your own encoding layer. You can follow along the code with these graphs.<br>
**(Good Luck! :P)**
<hr>

### MultiHeadAttention Mechanism (MHA)

![MHA](https://github.com/shuxg2017/transformer_encoder_package/blob/master/multi_head_attention_example/mha.PNG)
![QKV](https://github.com/shuxg2017/transformer_encoder_package/blob/master/multi_head_attention_example/mha_qkv_op.PNG)

<hr>

### Encoding Layer

![Enc](https://github.com/shuxg2017/transformer_encoder_package/blob/master/multi_head_attention_example/encoder_layer.PNG)
