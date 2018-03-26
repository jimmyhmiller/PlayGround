# instant-lambda

# Deploy

```shell
$ serverless deploy
```

# Redeploy Function

```
$ serverless deploy function -f echo
```

# Invoke

```shell
$ curl -X POST <url> -H 'Content-Type: application/json' -d '{"body": "Hi"}'
```
