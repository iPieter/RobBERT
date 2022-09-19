# Demo: 'die' vs 'dat' as a rest endpoint with Flask

As a demo, we release a small Flask example for a rest endpoint to analyse sentences. 
It will return whether a sentence contains the correct—according to RobBERT—use of 'die' or 'dat'.

By default, A Flask server will listen to port 5000. The endpoint is `/`.

## Get started
First install the dependencies from the requirements.txt file using `pip install -r requirements.txt`

```shell script
$ python app.py --model-path DTAI-KULeuven/robbertje-shuffled-dutch-die-vs-dat --fast-model-path pdelobelle/robbert-v2-dutch-base
```

## Classification model
Simply make a http POST request to `/` with the parameter `sentence` filled in:

```shell script
$ curl --data "sentence=Daar loopt _die_ meisje." localhost:5000
```

This should give you the following response:

```json
{
    "rating": 1, 
    "interpretation": "incorrect", 
    "confidence": 5.222124099731445, 
    "sentence": "Daar loopt _die_ meisje."
}
```

## Zero-shot model
We also have a faster zero-shot model (using RobBERT base), which might be faster and easier to use. There is a small drop in accuracy, but that should be quite limited.

To use the faster zero-shot model, just make a http POST request to `/fast` with the same parameter `sentence` filled in:

```shell script
$ curl --data "sentence=Daar loopt _die_ meisje." localhost:5000/fast
```

This should give you the following response, which is similar, but also provides `die`, `dat`, `Die` or `Dat` as the rating value:

```json
{
    "rating": "dat", 
    "interpretation": "incorrect", 
    "confidence": 2.567270278930664, 
    "sentence": "Daar loopt _die_ meisie."
}                       
```