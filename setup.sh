#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
NGINX_CONF_SRC="$REPO_DIR/nginx/api.ml.linguo.foo.conf"
NGINX_CONF_DST="/etc/nginx/sites-available/ml-api"
NGINX_ENABLED="/etc/nginx/sites-enabled/ml-api"

echo "=== linguo-ml setup ==="

# --- nginx ---
echo "[1/3] Настройка nginx..."
cp "$NGINX_CONF_SRC" "$NGINX_CONF_DST"
if [ ! -L "$NGINX_ENABLED" ]; then
    ln -s "$NGINX_CONF_DST" "$NGINX_ENABLED"
fi
nginx -t
systemctl reload nginx
echo "      nginx готов"

# --- .env ---
echo "[2/3] Проверка .env..."
if [ ! -f "$REPO_DIR/.env" ]; then
    cp "$REPO_DIR/.env.example" "$REPO_DIR/.env"
    echo "      .env создан из .env.example — заполни переменные!"
else
    echo "      .env уже существует, пропускаю"
fi

# --- venv + зависимости ---
echo "[3/3] Python venv и зависимости..."
if [ ! -d "$REPO_DIR/venv" ]; then
    python3 -m venv "$REPO_DIR/venv"
fi
"$REPO_DIR/venv/bin/pip" install --quiet --upgrade pip
"$REPO_DIR/venv/bin/pip" install --quiet -r "$REPO_DIR/requirements.txt"
echo "      venv готов"

echo ""
echo "=== Готово ==="
echo "Запуск сервера: venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000"
