# ai-lang AWS Lambda worker (Program B) as a container image.
#
# A Lambda container image brings its own userspace, so this Ubuntu base
# runs on Lambda directly (no Amazon-Linux glibc constraint, unlike a
# zip custom runtime). The build recipe below is exactly what is verified
# to compile and run on Linux x86_64: LLVM 21 via LLVM_SYS_211_PREFIX, the
# binary statically links LLVM and dlopens libcurl / libcrypto at runtime.
#
#   docker build -f deploy/Dockerfile -t ai-lang-lambda .
#   # then tag + push to ECR and create the function from the image:
#   #   aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
#   #   docker tag ai-lang-lambda <acct>.dkr.ecr.<region>.amazonaws.com/ai-lang-lambda:latest
#   #   docker push <acct>.dkr.ecr.<region>.amazonaws.com/ai-lang-lambda:latest
#   # Program A then does the CreateFunction (see deploy/README.md):
#   #   lambda_create_from_image(region, name, role, image_uri, ak, sk, token)

# ---- build stage ----
FROM ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl build-essential git \
        llvm-21 llvm-21-dev libpolly-21-dev \
        libcurl4-openssl-dev libssl-dev zlib1g-dev libzstd-dev \
    && rm -rf /var/lib/apt/lists/*
# Rust toolchain.
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
ENV LLVM_SYS_211_PREFIX=/usr/lib/llvm-21
WORKDIR /src
COPY . /src
RUN cargo build --release --bin ai-lang
# Bake the worker code into a content-addressed codebase.
RUN AI_LANG_CODEBASE=/build/codebase ./target/release/ai-lang add examples/lambda_handler.ail

# ---- runtime stage ----
FROM ubuntu:24.04
# Only the dynamic deps the binary actually needs (LLVM is static). The
# versioned-soname fallback in cffi.rs means we need no -dev packages:
# libcurl4 ships libcurl.so.4, libssl3 ships libcrypto.so.3.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libcurl4 libssl3 zlib1g libzstd1 libstdc++6 libffi8 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/ai-lang /opt/ai-lang/ai-lang
COPY --from=build /build/codebase /var/task/.ai-lang
COPY deploy/bootstrap /var/runtime/bootstrap
RUN chmod +x /var/runtime/bootstrap /opt/ai-lang/ai-lang
ENTRYPOINT ["/var/runtime/bootstrap"]
