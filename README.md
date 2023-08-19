# seq2seq transformer for machine translation
A PyTorch implementation of "Attention is all you need" by Vaswani et al

[Paper](https://arxiv.org/abs/1706.03762) </br>
[Dataset](http://www.manythings.org/anki/)

Sample outputs after (very) brief training:
```python
>>> inference("hello, how are you today?", model, dataset, DEV)
'<SOS> hola ¿cómo estás hoy <EOS>'

>>> inference("i like to play basketball in my free time", model, dataset, DEV)
'<SOS> me gusta jugar baloncesto en mi tiempo libre <EOS>'

>>> inference("i will go swim tomorrow", model, dataset, DEV)
'<SOS> mañana iré a nadar <EOS>'

>>> inference("i really want to sleep now", model, dataset, DEV)
'<SOS> realmente quiero dormir ahora <EOS>'

>>> inference("this took me a few days to make", model, dataset, DEV)
'<SOS> me tomó un par de días hacer esto <EOS>'
```
