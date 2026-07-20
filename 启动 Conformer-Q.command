#!/bin/zsh
set -euo pipefail

APP_DIR="${0:A:h}"
TOOLS_DIR="$APP_DIR/.tools"
URL="http://127.0.0.1:5062"

cd "$APP_DIR"

echo "Conformer-Q"
echo "项目目录: $APP_DIR"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo
  echo "此便携配置只支持 Apple Silicon Mac。当前系统: $(uname -s) $(uname -m)"
  echo "ORCA、CREST 和 xTB 都是 macOS arm64 可执行文件。"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo
    echo "未找到 python3。请先安装 Python/RDKit 环境后再启动。"
    exit 1
  fi
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import flask
from rdkit import Chem
PY
then
  echo
  echo "当前 Python 缺少 Flask 或 RDKit：$PYTHON_BIN"
  echo "请在你已有环境中安装 requirements.txt，或用 PYTHON_BIN 指定可用 Python。"
  echo "示例：PYTHON_BIN=/path/to/python ./启动 Conformer-Q.command"
  exit 1
fi

xattr -dr com.apple.quarantine "$APP_DIR/engines" "$APP_DIR/.tools" 2>/dev/null || true

ORCA_BIN="${CONFORMER_Q_ORCA_BIN:-}"
if [[ -z "$ORCA_BIN" && -n "${CONFORMER_Q_ORCA_DIR:-}" ]]; then
  ORCA_BIN="$CONFORMER_Q_ORCA_DIR/orca"
fi
if [[ -z "$ORCA_BIN" && -x "$APP_DIR/engines/orca-6.0.1/orca" ]]; then
  ORCA_BIN="$APP_DIR/engines/orca-6.0.1/orca"
fi
if [[ -z "$ORCA_BIN" && -x "$(command -v orca 2>/dev/null || true)" ]]; then
  ORCA_BIN="$(command -v orca)"
fi
if [[ -z "$ORCA_BIN" || ! -x "$ORCA_BIN" ]]; then
  echo
  echo "提示：未检测到 ORCA，精修功能不可用；可设置 CONFORMER_Q_ORCA_DIR 或 CONFORMER_Q_ORCA_BIN。"
fi

echo
echo "正在启动服务: $URL"
if [[ "${1:-}" == "--setup-only" ]]; then
  echo "依赖环境已可用：$PYTHON_BIN"
  exit 0
fi

(sleep 2; open "$URL") &
exec "$PYTHON_BIN" "$APP_DIR/app.py"
