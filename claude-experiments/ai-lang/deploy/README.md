# Deploying ai-lang to AWS Lambda

The whole loop is ai-lang: **Program A** drives the AWS control plane to
provision and invoke a Lambda; **Program B** runs *as* that Lambda. This
directory packages B so it actually cold-starts.

## What is proven

On Linux x86_64 (the Lambda target platform):

- The full test suite passes (`cargo test --lib`, 397 tests) — the C FFI
  (dlopen of system `libcurl`/`libcrypto`), SigV4, HTTP, S3, the
  zip/base64/crc32 packaging, the GC, `defer`, and the distributed runtime
  all work.
- The real `ai-lang` binary, run as `ai-lang run main`
  (`main = lambda_serve(|e| handler(e))`), behaves as a Lambda custom
  runtime: it polls the Runtime API, runs the handler, and posts the
  response. Verified against a Runtime API emulator
  (`examples/lambda_runtime_api_mock.py`).
- That same binary performs real AWS calls — a live S3 PUT/GET/DELETE
  round-trip (200/200/204).

So the deployable artifact (the ai-lang binary + a baked codebase + the
`bootstrap`) is a working Lambda worker. The container image just wraps
it.

## Program B — the worker

`examples/lambda_handler.ail`:

```
def handler(event: String) -> String =
    match json_field(event, "name") {
        Result::Ok(name) => str3("{\"greeting\":\"hello ", name, "\"}"),
        Result::Err(e)   => "{\"greeting\":\"hello world\"}"
    }

def main() -> Int = lambda_serve(|e: String| handler(e))
```

`bootstrap` runs `ai-lang run main`; `lambda_serve` reads
`AWS_LAMBDA_RUNTIME_API` (Lambda sets it) and serves forever.

## Build + push the image

```sh
docker build -f deploy/Dockerfile -t ai-lang-lambda .

REGION=us-east-1; ACCT=$(aws sts get-caller-identity --query Account --output text)
REPO=$ACCT.dkr.ecr.$REGION.amazonaws.com/ai-lang-lambda
aws ecr create-repository --repository-name ai-lang-lambda 2>/dev/null || true
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCT.dkr.ecr.$REGION.amazonaws.com
docker tag ai-lang-lambda $REPO:latest
docker push $REPO:latest
```

(A Lambda container image carries its own userspace, so the Ubuntu base
runs on Lambda as-is. The Dockerfile build is the verified Linux x86_64
recipe — LLVM 21 statically linked, `libcurl`/`libcrypto` dlopened at
runtime via the versioned-soname fallback, so no `-dev` packages are
needed in the runtime stage.)

## Program A — provision and invoke

From ai-lang (needs an execution role ARN and the image URI above):

```
let region = "us-east-1";
let image  = "<acct>.dkr.ecr.us-east-1.amazonaws.com/ai-lang-lambda:latest";

// create the function from the container image
let c = lambda_create_from_image(region, "prog-b", role_arn, image, ak, sk, token);

// invoke it
let r = lambda_invoke(region, "prog-b", "{\"name\":\"world\"}", ak, sk, token);
// -> Ok(HttpResponse { status: 200, body: "{\"greeting\":\"hello world\"}" })

// tear it down
let d = lambda_delete_function(region, "prog-b", ak, sk, token);
```

For a zip-based deploy instead of a container, A can build the package
itself (`zip_one`), upload it (`s3_put_object`), and
`lambda_create_from_s3` — though for a JIT runtime the container image is
the better fit.

`examples/aws_provision_demo.ail` runs create -> invoke -> delete, and
`examples/aws_lambda_demo.ail` / `examples/s3_demo.ail` exercise invoke
and S3 against real AWS.
