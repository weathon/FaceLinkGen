#!/usr/bin/env bash
set -euo pipefail

CSI=$'\033['
RST="${CSI}0m"
BOLD="${CSI}1m"
DIM="${CSI}2m"

GRN="${CSI}32m"
YEL="${CSI}33m"
RED="${CSI}31m"
CYN="${CSI}36m"
WHT="${CSI}37m"

host="${HOSTNAME:-node42}"
now="$(LC_ALL=C date '+%a %b %d %T %Y')"

p() { printf '%b\n' "$1"; }

p "${DIM}Every 1.0s: ${BOLD}nvidia-smi${RST}${DIM}  ${host}: ${now}${RST}"
p ""

p "${BOLD}+-----------------------------------------------------------------------------------------------+${RST}"
p "${BOLD}|${RST} NVIDIA-SMI ${CYN}560.12${RST}                  Driver Version: ${CYN}560.12${RST}            CUDA Version: ${CYN}12.5${RST}       ${BOLD}|${RST}"
p "${BOLD}|-----------------------------------------+-----------------------------+------------------------|${RST}"
p "${BOLD}|${RST} GPU  Name                    Persistence-M ${BOLD}|${RST} Bus-Id            Disp.A ${BOLD}|${RST} Volatile Uncorr. ECC ${BOLD}|${RST}"
p "${BOLD}|${RST} Fan  Temp   Perf   Pwr:Usage/Cap          ${BOLD}|${RST}           Memory-Usage   ${BOLD}|${RST} GPU-Util  Compute M. ${BOLD}|${RST}"
p "${BOLD}|${RST}                                         ${BOLD}|${RST}                            ${BOLD}|${RST}               MIG M. ${BOLD}|${RST}"
p "${BOLD}|=========================================+=============================+========================|${RST}"

gpu_line() {
  local id=$1 fan=$2 temp=$3 pwr=$4 mem=$5 util=$6 bus=$7
  p "${BOLD}|${RST}  ${id}  NVIDIA B200 SXM               On     ${BOLD}|${RST} ${bus}     Off ${BOLD}|${RST}                  Off ${BOLD}|${RST}"
  p "${BOLD}|${RST} ${YEL}${fan}%%${RST}   ${GRN}${temp}C${RST}    P0        ${YEL}${pwr}W${RST} / 1500W      ${BOLD}|${RST} ${mem}MiB / 245760MiB    ${BOLD}|${RST}  ${GRN}${util}%%${RST}      Default ${BOLD}|${RST}"
  p "${BOLD}|${RST}                                         ${BOLD}|${RST}                            ${BOLD}|${RST}                  N/A ${BOLD}|${RST}"
  p "${BOLD}+-----------------------------------------+-----------------------------+------------------------+${RST}"
}

gpu_line 0 62 71 1380 198432 99 00000000:01:00.0
gpu_line 1 64 73 1412 201120 100 00000000:02:00.0
gpu_line 2 60 70 1348 195876 98 00000000:03:00.0
gpu_line 3 63 72 1395 199540 99 00000000:04:00.0
gpu_line 4 61 71 1362 197210 99 00000000:05:00.0
gpu_line 5 65 74 1430 202345 100 00000000:06:00.0
gpu_line 6 59 69 1335 194880 97 00000000:07:00.0
gpu_line 7 63 72 1398 199998 99 00000000:08:00.0

p ""
p "${BOLD}+-----------------------------------------------------------------------------------------------+${RST}"
p "${BOLD}|${RST} Processes:                                                                                    ${BOLD}|${RST}"
p "${BOLD}|${RST}  GPU   GI   CI        PID   Type   Process name                                  GPU Memory   ${BOLD}|${RST}"
p "${BOLD}|${RST}        ID   ID                                                               Usage            ${BOLD}|${RST}"
p "${BOLD}|===============================================================================================|${RST}"

for i in {0..7}; do
  printf "%b\n" "${BOLD}|${RST}    ${i}   N/A  N/A   82430$((i+1))      C   python                                         $((195000 + (RANDOM % 8000)))MiB    ${BOLD}|${RST}"
done

p "${BOLD}+-----------------------------------------------------------------------------------------------+${RST}"
