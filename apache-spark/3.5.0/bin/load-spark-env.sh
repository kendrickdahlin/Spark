#!/bin/bash
JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home}" exec "/opt/homebrew/Cellar/apache-spark/3.5.0/libexec/bin/load-spark-env.sh"  "$@"
