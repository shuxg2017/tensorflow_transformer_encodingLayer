# Transformer Encoder
1. Build [transformer encoder package](https://github.com/shuxg2017/transformer_encoder_demo/blob/master/encoders/transformer_encoder.py).
2. Show how to use the package and visualize attention weights in [this notebook](https://github.com/shuxg2017/transformer_encoder_demo/blob/master/transformer_encoder_demo.ipynb).
3. Goal is to customize BERT, original BERT is too large.
4. You can modify [this package](https://github.com/shuxg2017/transformer_encoder_demo/blob/master/encoders/transformer_encoder.py).
5. Note: there are 2 classes, the codes of EncoderLayers and EncoderModel are the same, BUT
   - EncoderLayers inherits tf.keras.layers.Layer
   - EncoderModel inherits tf.keras.Model
