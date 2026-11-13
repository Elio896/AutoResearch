#!/usr/bin/env bash
# EvoResearch 阿里云一键部署
#
# 用法（在服务器上，项目根目录）：
#   bash deploy/aliyun_setup.sh
#
# 做的事：建 venv → 装依赖 → 写 systemd 服务 → 启动
# 注意：密钥只从环境变量读，**不写进任何文件**。

set -euo pipefail

APP_NAME="evoresearch"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-python3}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "=== EvoResearch deploy ==="
echo "app dir: $APP_DIR"
echo "python:  $($PY --version)"
echo "listen:  $HOST:$PORT"
echo

# 1) venv
if [ ! -d "$APP_DIR/.venv" ]; then
  echo "[1/4] creating venv"
  $PY -m venv "$APP_DIR/.venv"
else
  echo "[1/4] venv exists"
fi

# 2) deps
echo "[2/4] installing deps"
"$APP_DIR/.venv/bin/python" -m pip install --upgrade pip --quiet
if [ -f "$APP_DIR/requirements-api.txt" ]; then
  "$APP_DIR/.venv/bin/python" -m pip install -r "$APP_DIR/requirements-api.txt" --quiet
else
  "$APP_DIR/.venv/bin/python" -m pip install fastapi uvicorn scikit-learn pandas numpy --quiet
fi

# 3) systemd unit
echo "[3/4] writing systemd unit"
UNIT_PATH="/etc/systemd/system/${APP_NAME}.service"
sudo tee "$UNIT_PATH" > /dev/null <<UNIT
[Unit]
Description=EvoResearch API
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
# 密钥只放在 EnvironmentFile 里，且该文件权限必须是 600
EnvironmentFile=-/etc/${APP_NAME}.env
ExecStart=$APP_DIR/.venv/bin/python -m uvicorn api:create_app --factory --host $HOST --port $PORT
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

# 4) secrets file (if missing)
if [ ! -f "/etc/${APP_NAME}.env" ]; then
  echo "[4/4] creating empty env file (fill in your keys, then chmod 600)"
  sudo tee "/etc/${APP_NAME}.env" > /dev/null <<'ENV'
# 填完改成 600：sudo chmod 600 /etc/evoresearch.env
# DEEPSEEK_API_KEY=
# TYPESAFE_API_KEY=
ENV
  sudo chmod 600 "/etc/${APP_NAME}.env"
fi

sudo systemctl daemon-reload
sudo systemctl enable "$APP_NAME"
sudo systemctl restart "$APP_NAME"

echo
echo "=== deployed ==="
echo "status: sudo systemctl status $APP_NAME"
echo "logs:   sudo journalctl -u $APP_NAME -f"
echo "health: curl -s http://127.0.0.1:$PORT/health || true"
echo
echo "记得：Cloudflare/Nginx 反代 + HTTPS，别把这个端口直接暴露在公网。"
