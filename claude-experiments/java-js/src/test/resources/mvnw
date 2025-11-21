#!/bin/sh
# Maven Wrapper script

set -e

MAVEN_VERSION=3.9.6
BASE_URL="https://repo.maven.apache.org/maven2/org/apache/maven/apache-maven"
MAVEN_HOME="${HOME}/.m2/wrapper/apache-maven-${MAVEN_VERSION}"

if [ ! -d "${MAVEN_HOME}" ]; then
    echo "Downloading Maven ${MAVEN_VERSION}..."
    mkdir -p "${HOME}/.m2/wrapper"
    curl -L "${BASE_URL}/${MAVEN_VERSION}/apache-maven-${MAVEN_VERSION}-bin.tar.gz" | tar xz -C "${HOME}/.m2/wrapper"
fi

exec "${MAVEN_HOME}/bin/mvn" "$@"
