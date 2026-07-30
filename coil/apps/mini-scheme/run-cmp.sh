#!/usr/bin/env bash
# run a scheme program through mini-scheme AND chez, compare stdout
MS=/tmp/mini-scheme
prog="$1"
mine=$("$MS" < "$prog" 2>/tmp/gcstats)
# chez: wrap the last form in (display ...) — feed defines + (display last)
chez=$(chez --quiet 2>/dev/null <<CHEZ
$(sed '$d' "$prog")
(display $(tail -1 "$prog"))(newline)
CHEZ
)
gc=$(cat /tmp/gcstats)
if [ "$mine" = "$chez" ]; then echo "OK   $(basename $prog): $mine   $gc"; else echo "FAIL $(basename $prog): mini=[$mine] chez=[$chez]"; fi
