#!/bin/sh
set -eu

find src tests native scripts -type f \( -name '*.coil' -o -name '*.c' -o -name '*.h' -o -name '*.sh' \) \
  -exec wc -l {} + |
awk '$2 != "total" && $1 > 4000 {
       print $2 ": " $1 " lines exceeds the 4000-line limit" > "/dev/stderr"
       failed = 1
     }
     END { exit failed }'
