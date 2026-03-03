#!/bin/bash
set -e

cd "$(dirname "$0")"

STAGE="stages/stage0"
CP="$STAGE:asm.jar"

if [ ! -d "$STAGE" ]; then
    echo "ERROR: $STAGE does not exist. Run build.sh first to create the Java bootstrap, then:"
    echo "  mkdir -p $STAGE"
    echo "  java -cp 'asm.jar:build' jrust.Main compiler.jrs"
    echo "  cp output/*.class $STAGE/"
    exit 1
fi

echo "=== Compiling compiler.jrs with stage0 ==="
java -cp "$CP" Main compiler.jrs

echo "=== Checking fixed point ==="
DIFF=$(diff <(cd "$STAGE" && md5 -r *.class | sort) <(cd output && md5 -r *.class | sort) || true)

if [ -z "$DIFF" ]; then
    echo "Fixed point reached. No changes needed."
else
    echo "Output differs from stage0. Updating stage0..."
    rm -f "$STAGE"/*.class
    cp output/*.class "$STAGE"/

    echo "=== Verifying new stage0 reaches fixed point ==="
    java -cp "$CP" Main compiler.jrs

    DIFF2=$(diff <(cd "$STAGE" && md5 -r *.class | sort) <(cd output && md5 -r *.class | sort) || true)
    if [ -z "$DIFF2" ]; then
        echo "Fixed point confirmed."
    else
        echo "WARNING: New stage0 does NOT reach fixed point!"
        echo "$DIFF2"
        exit 1
    fi
fi

echo "=== Done ==="
