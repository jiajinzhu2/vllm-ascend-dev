#!/bin/bash
# ============================================================
# Dev Container 创建后初始化
#
# 由所有版本的 devcontainer postCreateCommand 共用，用于配置容器内通用开发环境。
# 本脚本已内联所需辅助函数和公司代理 CA 安装逻辑，不依赖工作区内其他脚本。
#
# 用法:
#   ./.devcontainer/post-create.sh             # 初始化 devcontainer
#   ./.devcontainer/post-create.sh -h | --help # 查看帮助
# ============================================================

set -euo pipefail
shopt -s nullglob

# ---- 默认配置 ----
DEFAULT_GIT_USER_NAME="jiajinzhu2"
DEFAULT_GIT_USER_EMAIL="jiajinzhu@huawei.com"
CORP_CA_TARGET_HOST="auth.openai.com"
CORP_CA_NAME="corp-proxy-ca"
CORP_CA_TMP_DIR=""

# ---- 公共函数 ----
log_error() {
    echo "[ERROR] $*" >&2
}

log_warn() {
    echo "[WARN] $*"
}

log_info() {
    echo "[INFO] $*"
}

log_ok() {
    echo "[OK] $*"
}

log_skip() {
    echo "  [SKIP] $*"
}

log_step() {
    echo ">>> $*"
}

command_exists() {
    command -v "$1" &>/dev/null
}

require_commands() {
    local cmd
    for cmd in "$@"; do
        if ! command_exists "$cmd"; then
            log_error "缺少依赖: $cmd"
            exit 1
        fi
    done
}

cleanup_corp_ca_tmp_dir() {
    if [[ -n "$CORP_CA_TMP_DIR" && -d "$CORP_CA_TMP_DIR" ]]; then
        rm -rf -- "$CORP_CA_TMP_DIR"
    fi
}

trap cleanup_corp_ca_tmp_dir EXIT

# ---- 参数解析 ----
print_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "初始化 devcontainer 内的通用开发环境。"
    echo ""
    echo "选项:"
    echo "  -h, --help  显示此帮助信息"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) print_help ;;
        *) log_error "未知参数: $1，使用 -h 查看帮助"; exit 1 ;;
    esac
done

# ---- 环境检查 ----
require_commands git sed

# ---- 主逻辑 ----
ensure_agent_config_dirs() {
    log_step "初始化 AI Agent 配置目录..."
    mkdir -p "$HOME/.codex" "$HOME/.claude"
    log_ok "$HOME/.codex 和 $HOME/.claude 已就绪"
}

fix_atb_env() {
    log_step "修正 Ascend ATB 环境配置..."
    local env_file
    local changed=false

    for env_file in /root/.bashrc /etc/profile; do
        if [[ ! -f "$env_file" ]]; then
            log_skip "$env_file 不存在"
            continue
        fi

        sed -i -E 's#^[[:space:]]*source /usr/local/Ascend/nnal/atb/set_env\.sh([[:space:]]+--cxx_abi=[01])?[[:space:]]*$#source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0#' "$env_file"
        log_ok "$env_file 已检查"
        changed=true
    done

    if [[ "$changed" == false ]]; then
        log_skip "未发现需要检查的环境文件"
    fi
}

install_corp_ca() {
    log_step "安装公司代理 CA..."

    if [[ $EUID -ne 0 ]]; then
        log_error "需要 root 权限安装系统 CA，请用 root 运行或加 sudo"
        exit 1
    fi

    local proxy="${https_proxy:-${HTTPS_PROXY:-}}"
    if [[ -z "$proxy" ]]; then
        log_error "未获取到代理地址，请设置 \$https_proxy 或 \$HTTPS_PROXY"
        exit 1
    fi

    local proxy_host_port="${proxy#http://}"
    proxy_host_port="${proxy_host_port#https://}"
    proxy_host_port="${proxy_host_port%%/*}"

    require_commands openssl curl awk grep mktemp

    local ca_store
    local ca_file
    if command_exists update-ca-certificates; then
        ca_store="debian"
        ca_file="/usr/local/share/ca-certificates/${CORP_CA_NAME}.crt"
    elif command_exists update-ca-trust; then
        ca_store="rhel"
        ca_file="/etc/pki/ca-trust/source/anchors/${CORP_CA_NAME}.crt"
    else
        log_error "缺少系统 CA 刷新命令: update-ca-certificates 或 update-ca-trust"
        echo "  Debian/Ubuntu: apt install ca-certificates"
        echo "  RHEL/CentOS/openEuler: yum install ca-certificates"
        exit 1
    fi

    if curl -sS -o /dev/null --max-time 10 --proxy "$proxy" \
            "https://$CORP_CA_TARGET_HOST/" 2>/dev/null; then
        log_skip "系统已信任公司代理 CA，无需重装"
        return
    fi

    CORP_CA_TMP_DIR="$(mktemp -d)"

    log_step "抓取公司代理 MITM 证书链 ($CORP_CA_TARGET_HOST via $proxy_host_port)..."
    if ! echo | openssl s_client -proxy "$proxy_host_port" \
            -connect "$CORP_CA_TARGET_HOST:443" \
            -servername "$CORP_CA_TARGET_HOST" -showcerts \
            2>/dev/null > "$CORP_CA_TMP_DIR/chain.pem"; then
        log_error "openssl 抓取证书链失败（确认 openssl 支持 -proxy 且代理可达）"
        exit 1
    fi
    if [[ ! -s "$CORP_CA_TMP_DIR/chain.pem" ]] || \
            ! grep -q 'BEGIN CERTIFICATE' "$CORP_CA_TMP_DIR/chain.pem"; then
        log_error "未获取到任何证书"
        exit 1
    fi

    awk -v d="$CORP_CA_TMP_DIR" \
        'BEGIN{n=0} /-----BEGIN CERTIFICATE-----/{n++} {print > d"/cert_"n".pem"}' \
        "$CORP_CA_TMP_DIR/chain.pem"

    local root_cert=""
    local cert_file
    local subject
    local issuer
    for cert_file in "$CORP_CA_TMP_DIR"/cert_*.pem; do
        [[ -s "$cert_file" ]] || continue
        subject="$(openssl x509 -in "$cert_file" -noout -subject 2>/dev/null | sed 's/^subject=//' || true)"
        issuer="$(openssl x509 -in "$cert_file" -noout -issuer 2>/dev/null | sed 's/^issuer=//' || true)"
        if [[ -n "$subject" && "$subject" == "$issuer" ]]; then
            root_cert="$cert_file"
            break
        fi
    done
    if [[ -z "$root_cert" ]]; then
        log_error "证书链中未找到自签根 CA（代理可能未下发根证书）"
        exit 1
    fi
    log_info "找到自签根 CA: $(openssl x509 -in "$root_cert" -noout -subject 2>/dev/null)"

    log_step "安装到 $ca_file ..."
    mkdir -p "$(dirname "$ca_file")"
    cp "$root_cert" "$ca_file"
    chmod 644 "$ca_file"

    local update_output
    case "$ca_store" in
        debian)
            update_output="$(update-ca-certificates 2>&1 | grep -m1 -E '[0-9]+ added, [0-9]+ removed' || true)"
            [[ -z "$update_output" ]] && update_output="done"
            log_ok "update-ca-certificates: $update_output"
            ;;
        rhel)
            if ! update_output="$(update-ca-trust extract 2>&1)"; then
                log_error "update-ca-trust extract 失败"
                [[ -n "$update_output" ]] && echo "$update_output"
                exit 1
            fi
            [[ -z "$update_output" ]] && update_output="done"
            log_ok "update-ca-trust extract: $update_output"
            ;;
    esac

    log_step "验证 TLS 信任..."
    if curl -sS -o /dev/null --max-time 15 --proxy "$proxy" \
            "https://$CORP_CA_TARGET_HOST/" 2>/dev/null; then
        log_ok "公司代理 CA 已安装并信任"
    else
        log_error "安装后验证仍失败，请检查代理与证书链"
        exit 1
    fi

    cleanup_corp_ca_tmp_dir
    CORP_CA_TMP_DIR=""
}

configure_git_proxy() {
    log_step "配置 Git 代理..."
    local git_proxy="${devcontainer_proxy:-}"

    if [[ -z "$git_proxy" ]]; then
        git_proxy="${https_proxy:-${HTTPS_PROXY:-}}"
    fi

    if [[ -z "$git_proxy" ]]; then
        git_proxy="${http_proxy:-${HTTP_PROXY:-}}"
    fi

    if [[ -z "$git_proxy" ]]; then
        log_skip "未设置 devcontainer_proxy / http_proxy / https_proxy"
        return
    fi

    git config --global http.proxy "$git_proxy"
    log_ok "git http.proxy 已配置"
}

configure_git_identity() {
    log_step "配置 Git 用户信息..."
    local git_user_name="${devcontainer_git_user_name:-${DEVCONTAINER_GIT_USER_NAME:-$DEFAULT_GIT_USER_NAME}}"
    local git_user_email="${devcontainer_git_user_email:-${DEVCONTAINER_GIT_USER_EMAIL:-$DEFAULT_GIT_USER_EMAIL}}"

    if [[ -z "$git_user_name" || -z "$git_user_email" ]]; then
        log_skip "未设置 Git user.name / user.email"
        return
    fi

    git config --global user.name "$git_user_name"
    git config --global user.email "$git_user_email"
    log_ok "git user.name / user.email 已配置"
}

configure_pip_index() {
    log_step "配置 pip 镜像..."

    if ! command_exists pip; then
        log_warn "pip 未找到，跳过 pip 镜像配置"
        return
    fi

    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    log_ok "pip index-url 已配置"
}

ensure_agent_config_dirs
fix_atb_env
install_corp_ca
configure_git_proxy
configure_git_identity
configure_pip_index

log_ok "devcontainer post-create 初始化完成"
