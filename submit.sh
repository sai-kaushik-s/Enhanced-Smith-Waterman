#!/usr/bin/env bash
set -euo pipefail

TEAM=${1:-2025CSZ8470_2025ANZ8223}
OUTPUT=submit_${TEAM}_$(date +%Y%m%d_%H%M%S).tar.gz

rm *${TEAM}*gz || true

tar -czf ${OUTPUT} README.md baseline optimized Makefile run.sh submit.sh setup.sh benchmark.py plots
echo "Created submission package: ${OUTPUT}"
echo "Please upload ${OUTPUT} in Moodle "
