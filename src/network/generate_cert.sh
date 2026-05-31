#!/usr/bin/env bash
# Generate a self-signed TLS 1.3 cert for the inference server.
#
# Usage:
#   ./src/network/generate_cert.sh
#   ./src/network/generate_cert.sh GROUND_STATION_IP
#
# The resulting server.crt + server.key go into network/certs/.
# Copy server.crt to the edge so it can verify the server's TLS chain.

set -euo pipefail

IP_OR_HOST="${1:-127.0.0.1}"
CN="ground-station.local"
DAYS=365

CERT_DIR="$(dirname "$0")/certs"
mkdir -p "$CERT_DIR"

# If the argument looks like an IP, use IP:; otherwise use DNS:.
if [[ "$IP_OR_HOST" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    SAN="DNS:${CN},DNS:localhost,IP:${IP_OR_HOST},IP:127.0.0.1"
else
    SAN="DNS:${CN},DNS:${IP_OR_HOST},DNS:localhost,IP:127.0.0.1"
fi

echo "Generating self-signed cert"
echo "  CN:          ${CN}"
echo "  SAN:         ${SAN}"
echo "  Validity:    ${DAYS} days"
echo "  Output dir:  ${CERT_DIR}"
echo

openssl req -x509 -newkey rsa:4096 -nodes \
    -keyout "${CERT_DIR}/server.key" \
    -out    "${CERT_DIR}/server.crt" \
    -days   "${DAYS}" \
    -subj   "/CN=${CN}" \
    -addext "subjectAltName=${SAN}"

chmod 600 "${CERT_DIR}/server.key"

echo
echo "Done. Files:"
ls -la "${CERT_DIR}"
echo
echo "To trust this cert on the edge client, copy ${CERT_DIR}/server.crt to"
echo "the same path on the Jetson and point the client at it via --cafile."
