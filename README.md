# Backend

## Usage

In your terminal, run:
```
$ export FLASK_APP=flask_app.py
$ flask run --host=0.0.0.0 --port=8000
```
Then, your frontend will be able to request the APIs.

The frontend will request on troubadour:8000

## Notes

Grobid web service is currently running on troubadour:8081 using docker.

If the docker is down (due to frequently restart of server = =), restart it with:
```
$ docker run -t --rm --init -p 8081:8070 -p 8082:8071 lfoppiano/grobid:0.6.1
```
