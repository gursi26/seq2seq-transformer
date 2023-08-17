# seq2seq transformer for machine translation
A PyTorch implementation of "Attention is all you need" by Vaswani et al

[Paper](https://arxiv.org/abs/1706.03762) </br>
[Dataset](http://www.manythings.org/anki/)

Sample outputs after (very) brief training:
```python
>>> inference("hello, how are you today", model, dataset, DEV, 50)
'<SOS> hola ¿cómo hola estás hoy <EOS>'

>>> inference("i like to play tennis in my free time", model, dataset, DEV, 50)
'<SOS> me gusta jugar al tenis libre me gusta jugar tenis en mis tiempo libre <EOS>'

>>> inference("i really want to sleep now", model, dataset, DEV, 50)
'<SOS> realmente quiero dormir ahora me quiero dormir con la rompió hasta las cartas <EOS>'
```
