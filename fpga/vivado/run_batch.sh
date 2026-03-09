#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VIVADO_BIN="${VIVADO_BIN:-vivado}"
VIVADO_BAT_WIN="${VIVADO_BAT_WIN:-}"
TCL_SCRIPT="${SCRIPT_DIR}/run_batch.tcl"

RUN_DIR=""
PART="xc7a35tcpg236-1"
TOP="top_level"
XDC="${REPO_ROOT}/CNN_constraints.xdc"
CLOCK_PERIOD_NS=""
JOBS="4"
GENERICS=()

usage() {
  cat <<USAGE
Usage:
  fpga/vivado/run_batch.sh --run-dir <dir> [options]

Options:
  --run-dir <dir>           Required run output directory (under results/)
  --repo-root <dir>         Repo root (default: auto-detected)
  --part <part>             FPGA part (default: ${PART})
  --top <module>            Top module (default: ${TOP})
  --xdc <path>              XDC path (default: ${XDC})
  --clock-period-ns <ns>    Target clock period in ns (optional)
  --jobs <n>                Number of parallel jobs (default: ${JOBS})
  --generic KEY=VALUE       Verilog generic override (repeatable)
  --tcl <path>              Tcl script path (default: fpga/vivado/run_batch.tcl)
                            WSL note: set VIVADO_BAT_WIN to a Windows vivado.bat path
  -h, --help                Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"; shift 2 ;;
    --repo-root)
      REPO_ROOT="$2"; shift 2 ;;
    --part)
      PART="$2"; shift 2 ;;
    --top)
      TOP="$2"; shift 2 ;;
    --xdc)
      XDC="$2"; shift 2 ;;
    --clock-period-ns)
      CLOCK_PERIOD_NS="$2"; shift 2 ;;
    --jobs)
      JOBS="$2"; shift 2 ;;
    --generic)
      GENERICS+=("$2"); shift 2 ;;
    --tcl)
      TCL_SCRIPT="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "${RUN_DIR}" ]]; then
  echo "ERROR: --run-dir is required" >&2
  usage
  exit 2
fi

if [[ ! -f "${TCL_SCRIPT}" ]]; then
  echo "ERROR: TCL script not found: ${TCL_SCRIPT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${RUN_DIR}")"
RUN_DIR="$(cd "$(dirname "${RUN_DIR}")" && pwd)/$(basename "${RUN_DIR}")"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/reports"

if [[ "${XDC}" != /* ]]; then
  XDC="${REPO_ROOT}/${XDC}"
fi

GENERIC_CSV=""
if [[ ${#GENERICS[@]} -gt 0 ]]; then
  GENERIC_CSV="$(IFS=,; echo "${GENERICS[*]}")"
fi

if [[ -n "${VIVADO_BAT_WIN}" ]]; then
  if ! command -v cmd.exe >/dev/null 2>&1; then
    echo "ERROR: cmd.exe not found; required when VIVADO_BAT_WIN is set." >&2
    exit 1
  fi
  if ! command -v wslpath >/dev/null 2>&1; then
    echo "ERROR: wslpath not found; required when VIVADO_BAT_WIN is set." >&2
    exit 1
  fi

  TCL_SCRIPT_WIN="$(wslpath -w "${TCL_SCRIPT}")"
  REPO_ROOT_WIN="$(wslpath -w "${REPO_ROOT}")"
  RUN_DIR_WIN="$(wslpath -w "${RUN_DIR}")"
  XDC_WIN="$(wslpath -w "${XDC}")"

  exec cmd.exe /C "cd /d \"${RUN_DIR_WIN}\" && \"${VIVADO_BAT_WIN}\" -mode batch -source \"${TCL_SCRIPT_WIN}\" -tclargs \"${REPO_ROOT_WIN}\" \"${RUN_DIR_WIN}\" \"${PART}\" \"${TOP}\" \"${XDC_WIN}\" \"${CLOCK_PERIOD_NS}\" \"${JOBS}\" \"${GENERIC_CSV}\""
fi

if ! command -v "${VIVADO_BIN}" >/dev/null 2>&1; then
  echo "ERROR: vivado not found (set VIVADO_BIN or update PATH)." >&2
  exit 1
fi

exec "${VIVADO_BIN}" -mode batch -source "${TCL_SCRIPT}" \
  -tclargs "${REPO_ROOT}" "${RUN_DIR}" "${PART}" "${TOP}" "${XDC}" "${CLOCK_PERIOD_NS}" "${JOBS}" "${GENERIC_CSV}"
