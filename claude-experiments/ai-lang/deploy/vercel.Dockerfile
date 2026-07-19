# Build the ai-lang binary for the Vercel Functions runtime.
#
# Vercel Fluid runs the binary BARE on Amazon Linux 2023 (glibc 2.34), not
# in a container — so unlike deploy/Dockerfile (Ubuntu base, fine for a
# Lambda *container*), we must build against AL2023's glibc. We build ON
# amazonlinux:2023 for an exact match, with LLVM 21 from the official
# release tarball (AL2023 dnf only has up to llvm20).
#
# Extract the binary:
#   docker buildx build --platform linux/amd64 -f deploy/vercel.Dockerfile \
#     --target artifact -o type=local,dest=deploy/vercel-out .
# -> deploy/vercel-out/ai-lang  (x86_64, AL2023-glibc)

FROM --platform=linux/amd64 docker.io/amazonlinux:2023 AS build
RUN dnf -y install gcc gcc-c++ make git tar xz which findutils \
        zlib-devel libzstd-devel libffi-devel libxml2-devel ncurses-devel \
    && dnf clean all
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
# LLVM 21 (matches llvm-sys 211 / LLVM_SYS_211_PREFIX). Official release
# tarball; AL2023 dnf tops out at llvm20. The full dist is ~6GB extracted
# (clang/mlir/lldb/flang), which overflows the build disk — so stream it and
# extract ONLY what llvm-sys links against: static libs, headers, llvm-config.
RUN mkdir -p /opt/llvm \
    && curl -fL https://github.com/llvm/llvm-project/releases/download/llvmorg-21.1.8/LLVM-21.1.8-Linux-X64.tar.xz \
       | tar -xJ --strip-components=1 -C /opt/llvm --wildcards \
           '*/lib/*' '*/include/*' '*/bin/llvm-config'
# The release's `llvm-config --system-libs` emits absolute static-lib paths
# (e.g. /usr/lib/x86_64-linux-gnu/libzstd.a) from the box LLVM was built on,
# which llvm-sys's build.rs can't parse and which don't exist here. Wrap
# llvm-config to rewrite `…/libNAME.a` tokens to `-lNAME` (resolved against
# AL2023's own libs).
RUN mv /opt/llvm/bin/llvm-config /opt/llvm/bin/llvm-config.real \
    && printf '#!/bin/sh\n/opt/llvm/bin/llvm-config.real "$@" | sed -E "s#/[^ ]*/lib([a-zA-Z0-9_]+)\\.a#-l\\1#g"\n' \
       > /opt/llvm/bin/llvm-config \
    && chmod +x /opt/llvm/bin/llvm-config
ENV LLVM_SYS_211_PREFIX=/opt/llvm
WORKDIR /src
COPY . /src
RUN cargo build --release --bin ai-lang

# Export just the binary.
FROM scratch AS artifact
COPY --from=build /src/target/release/ai-lang /ai-lang
