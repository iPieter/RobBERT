# Demo: 'die' vs 'dat' as a rest endpoint with Flask

As a demo, we release a small Flask example for a rest endpoint to analyse sentences. 
It will return whether a sentence contains the correct---according to RobBERT---use of 'die' or 'dat'.

By default, A Flask server will listen to port 5000. The endpoint is `/`.

## Get started
First install the dependencies from the requirements.txt file using `pip install -r requirements.txt`

```shell script
$ python app.py
```

And then make a http POST request to `/` with the parameter `sentence` filled in:

```shell script
$ curl --data "sentence=Daar loopt _die_ meisje_." localhost:5000
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