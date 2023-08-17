# seq2seq transformer for machine translation
A PyTorch implementation of "Attention is all you need" by Vaswani et al

[Paper](https://arxiv.org/abs/1706.03762) </br>
[Dataset](http://www.manythings.org/anki/)

Sample outputs after (very) brief training:
```python
>>> inference("hello, how are you", model, dataset, DEV, 50)
'<SOS> hola ¿cómo estás hola ¿cómo estás hola hola hola hola ¿cómo estás hola <EOS>'

>>> inference("i like to play tennis in my free time", model, dataset, DEV, 50)
'<SOS> me gusta jugar al tenis libre en mi tiempo libre libre libre libre libre en mi tiempo libre <EOS>'

>>> inference("this took very long to make", model, dataset, DEV, 50)
'<SOS> esto llevó mucho tiempo en hacer esto para preparar mucho tiempo de hacerlo por hacer esto <EOS>'

>>> inference("i really want to sleep now", model, dataset, DEV, 50)
'<SOS> ahora quiero dormir en realidad quiero dormir profundamente quiero dormir con realmente dormir <EOS>'
```
