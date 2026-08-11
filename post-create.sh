#!/bin/bash
# ============================================================
# Dev Container 创建后初始化
#
# 由所有版本的 devcontainer postCreateCommand 共用，用于配置容器内通用开发环境。
#
# 用法:
#   ./.devcontainer/post-create.sh             # 初始化 devcontainer
#   ./.devcontainer/post-create.sh -h | --help # 查看帮助
# ============================================================

set -euo pipefail
# shellcheck source=../scripts/lib/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts/lib" && pwd)/common.sh"
ws_enter_workspace

# ---- 默认配置 ----
DEFAULT_GIT_USER_NAME="leolee"
DEFAULT_GIT_USER_EMAIL="yihao.li@huawei.com"

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
        *) ws_log_error "未知参数: $1，使用 -h 查看帮助"; exit 1 ;;
    esac
done

# ---- 环境检查 ----
ws_require_commands git sed

# ---- 主逻辑 ----
ensure_agent_config_dirs() {
    ws_log_step "初始化 AI Agent 配置目录..."
    mkdir -p "$HOME/.codex" "$HOME/.claude"
    ws_log_ok "~/.codex 和 ~/.claude 已就绪"
}

fix_atb_env() {
    ws_log_step "修正 Ascend ATB 环境配置..."
    local env_file
    local changed=false

    for env_file in /root/.bashrc /etc/profile; do
        if [[ ! -f "$env_file" ]]; then
            ws_log_skip "$env_file 不存在"
            continue
        fi

        sed -i -E 's#^[[:space:]]*source /usr/local/Ascend/nnal/atb/set_env\.sh([[:space:]]+--cxx_abi=[01])?[[:space:]]*$#source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0#' "$env_file"
        ws_log_ok "$env_file 已检查"
        changed=true
    done

    if [[ "$changed" == false ]]; then
        ws_log_skip "未发现需要检查的环境文件"
    fi
}

install_corp_ca() {
    ws_log_step "安装公司代理 CA..."
    bash "$SCRIPT_DIR/scripts/install-corp-ca.sh"
}

configure_git_proxy() {
    ws_log_step "配置 Git 代理..."
    local git_proxy="${devcontainer_proxy:-}"

    if [[ -z "$git_proxy" ]]; then
        git_proxy="${https_proxy:-${HTTPS_PROXY:-}}"
    fi

    if [[ -z "$git_proxy" ]]; then
        git_proxy="${http_proxy:-${HTTP_PROXY:-}}"
    fi

    if [[ -z "$git_proxy" ]]; then
        ws_log_skip "未设置 devcontainer_proxy / http_proxy / https_proxy"
        return
    fi

    git config --global http.proxy "$git_proxy"
    ws_log_ok "git http.proxy 已配置"
}

configure_git_identity() {
    ws_log_step "配置 Git 用户信息..."
    local git_user_name="${devcontainer_git_user_name:-${DEVCONTAINER_GIT_USER_NAME:-$DEFAULT_GIT_USER_NAME}}"
    local git_user_email="${devcontainer_git_user_email:-${DEVCONTAINER_GIT_USER_EMAIL:-$DEFAULT_GIT_USER_EMAIL}}"

    if [[ -z "$git_user_name" || -z "$git_user_email" ]]; then
        ws_log_skip "未设置 Git user.name / user.email"
        return
    fi

    git config --global user.name "$git_user_name"
    git config --global user.email "$git_user_email"
    ws_log_ok "git user.name / user.email 已配置"
}

configure_pip_index() {
    ws_log_step "配置 pip 镜像..."

    if ! ws_command_exists pip; then
        ws_log_warn "pip 未找到，跳过 pip 镜像配置"
        return
    fi

    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    ws_log_ok "pip index-url 已配置"
}

ensure_agent_config_dirs
fix_atb_env
install_corp_ca
configure_git_proxy
configure_git_identity
configure_pip_index

ws_log_ok "devcontainer post-create 初始化完成"
