# Text Generation with LSTM

Source: http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/


## proj-basic

### epoch = 50

* GPU util: 30%
* Mem usage: 10.3G
* Time per epoch: 398s, 180s, 464s, 467s, 462s, etc.
* Run time: ~7 hours
* **Best: epoch 49, loss: 1.2534**

Seed:
```
hon replied rather impatiently:
‘any shrimp could have told you that.’

‘if i’d been the whiting,’ s
```

Result:
```
aid the mock turtle. ‘but it would be all the season it it it in a bancee of the same torn the sea.’

‘i don’t shink they must brme,’ she gatter went on, ‘i don’t shink i can mays go out of the sea, to the mosse of the srees as tell me the door, and the doom bnd the pueen said the sabbit as the puppy and the sabbit had so done her voice, ‘it’s all the queen, and the morse of the sea--’

‘then it may so goow any rate,’ said the mock turtle. ‘but it would be all the season it it it in a bancee of the same torn the sea.’

‘i don’t shink they must brme,’ she gatter went on, ‘i don’t shink i can mays go out of the sea, to the mosse of the srees as tell me the door, and the doom bnd the pueen said the sabbit as the puppy and the sabbit had so done her voice, ‘it’s all the queen, and the morse of the sea--’

‘then it may so goow any rate,’ said the mock turtle. ‘but it would be all the season it it it in a bancee of the same torn the sea.’

‘i don’t shink they must brme,’ she gatter went on,
```


### epoch = 100

* GPU Util: 40%
* Run time: 42012 secs = ~11.67 hrs
* **Best:  epoch 70, loss: 1.2341**


Seed:
```
" ied to curtsey as she spoke--fancy curtseying as you’re falling
through the air! do you think you co "
```

Result:
```
uldn’t be no more that i could the larce harder that danst was the serpen of the sea, the was a little bll ouer their shares, and she thought the had not a very sook as it was a louse of the sharp cegan, and she shought the gad never seen the way the could, and she shought the had not a very sook as it was a dreath of the shapp bhind and the little golden key and she was so suedk to see it was in the sight hardeners, and the thened of her swrprise that the was a little startled about it, and the thing to tay the was of sight, and the there she was a little startled bll the court, but she was so much as she could see that it was oot and the roon all the white rabbit as eer of the should hard the court, but she was not a mongnt then and was sooething to see it to the shoee gardeners, and the there she was a little startled bll the court, but she was so much as she could see that it was oot and the roon all the white rabbit as eer of the should hard the court, but she was not a mongnt the
```

### epoch = 200

* Runtime: 80992.93805932999 secs
* **Best: epoch 60, loss: 1.2428** (This is WORSE than epoch = 100)

Seed:
```
" interrupt again. i
dare say there may be one.’
```

Result:
```
‘one, indeed!’ said the dormouse indignantly. howeve "
r, ‘what ase you the door with the babk and the sea, the doos up in the sea. the matter was an old coub in the sight had talking of the surprise of the sharp bhind bnd the thing to be alice foun, and the thing to tay the was of sight, and the thened of her seach in the soot as it wpuld be then she was a little gooders to see that it might be oueer to speak and shen there was no time to be talk about it; and she was a little startled all the court, but she was so much as she could see it to toat that harpen the way of sertlny, and she shought it was the pight as ie spokt of tertent, and the there she was a little startled bll to be talking to see it would be the cabk as herself about ier head in the distance,
and she was so much and shered in the soog.

‘what is the sea,’ the ming said to the jury, and the thought it was the pight as it were the coor with the tort of the thould hard the canee of the suoe. ‘i dear with they like the darch.  ie was sooe mine a long simence.’

‘i co wonder
```
