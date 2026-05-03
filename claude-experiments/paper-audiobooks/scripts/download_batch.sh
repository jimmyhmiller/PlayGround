#!/usr/bin/env bash
# Download 20 papers from jimmyhmiller's readings into samples/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p samples

declare -a papers=(
  "turing-computing-machinery|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/foc-covered/80051a11ac4916827dc3ffcfe90a3f206a6b016344fe1c2291bbd776fa83c6ee.pdf"
  "bush-as-we-may-think|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/foc-covered/90a39f10224b35c5cadddbe5c5722fd1d109f813c6059d10d9de2b817eeadcf4.pdf"
  "brooks-no-silver-bullet|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/foc-covered/fcbd318dbaa31c5b14b782b02ae25f53fc8aa33779d18ba1b4f126994c99b9ae.pdf"
  "brooks-mythical-man-month|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/foc-covered/6974bb56d2d9c0faf542073acc080d1420337c231ac534da734537dcad8ecc9a.pdf"
  "lamport-time-clocks|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/foc-covered/c55e7cab4230aa3d7126748a149b2db6f0d7a67296d5eccfdd50a210299a96b2.pdf"
  "engelbart-augmenting-human-intellect|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/foc-covered/c565fd479bd45f128ef165a9810fa48c9894eefee47b1e1bf8adc28d8d42ebe5.pdf"
  "licklider-man-computer-symbiosis|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/foc-covered/544ccc824d052ff4c9409dc3dd3f1c2834a72b7901146814368c9fb06ce737dc.pdf"
  "raymond-cathedral-bazaar|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/foc-potential/7792be4e1c7f9aaea480759f28ee2c1fcab53e3aac76a2b619ed9b2e4eab83e9.pdf"
  "hughes-why-functional-programming-matters|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/foc-potential/019dce2b82b7990794f5d63209c6d3bcec244d58148be1da50d4f697d776976e.pdf"
  "wadler-propositions-as-types|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/foc-covered/6316909c6a34ed1d7ce7bbf3bd191abaf738489ec73f517c9f579a9079c4be83.pdf"
  "backus-can-programming-be-liberated|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/foc-potential/2e412a7986c7fd1bdd060d90d035faa9ecc8a29675f99206c243da90954ca423.pdf"
  "hoare-emperors-old-clothes|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/foc-potential/7f1b4cacb5eccd86f5a400e3efced1fab577e8419711e5562a6f0a078ee819c2.pdf"
  "kay-personal-dynamic-media|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/foc-covered/3e6e2292ac60a80392e70fac907867f201b8c502bff578b451b3f4e98fad9c0e.pdf"
  "iverson-notation-tool-of-thought|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/foc-potential/8c98830e353aa12bc88754c0e793ddeee7238d7e23d9dc14401dfefe4815e4dc.pdf"
  "kay-real-computer-revolution|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/foc-potential/f11d5ac16d0e0a4b9ce9cc284aad20b7abf2f9f1033d96da3fd588a1f7ea0d0a.pdf"
  "hoare-axiomatic-basis|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/historical/cbc21ef34f54ca8f30b7f25a1486d8597726633a844692722dadc852e42c5b7f.pdf"
  "clark-darpa-internet-protocols|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/historical/eaca9db4be64b66f0aa237ef843fcd80605d63859ec1db725bfa16367e32ecf4.pdf"
  "ritchie-development-of-c|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/historical/c6641f097795aed588520dcc54f239aaa593eea72a8203a444fbdbb5dd66a9ff.pdf"
  "mccarthy-recursive-functions|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/historical/3d981849e59505eff3f14397a177b409f5d978d43d114bdd67c956e74320fc92.pdf"
  "dijkstra-humble-programmer|https://jimmyhmiller-bucket.s3.amazonaws.com/pdfs/foc-potential/f1f6b3cc137e405807479228c360e62a4872c2d44c1e9093bb1c2262175ecf92.pdf"
)

for entry in "${papers[@]}"; do
  name="${entry%%|*}"
  url="${entry#*|}"
  out="samples/${name}.pdf"
  if [[ -f "$out" ]]; then
    echo "[skip] $name (exists)"
    continue
  fi
  echo "[get]  $name"
  curl -sSLfo "$out" "$url"
done

echo "---"
ls -la samples/*.pdf | wc -l
echo "PDFs in samples/"
