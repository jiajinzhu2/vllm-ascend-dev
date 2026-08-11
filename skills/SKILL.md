---
name: setup-vllm-ascend-dev
description: Install, rebuild, repair, or verify an editable vLLM and vLLM Ascend development environment using the vLLM commit verified by vLLM Ascend main. Use for devcontainer rebuilds, matching vLLM to vLLM Ascend, removing incompatible torch/license/triton-ascend declarations, or troubleshooting this source installation.
---

# Set up vLLM Ascend development

Use `scripts/install-vllm-ascend.sh` for the deterministic workflow. It:

1. Clones or updates `vllm-project/vllm` and `vllm-project/vllm-ascend`.
2. Reads `.github/vllm-main-verified.commit` from vLLM Ascend.
3. Checks out that exact vLLM commit.
4. Applies the local packaging compatibility edits idempotently.
5. Installs the `setuptools_rust` build prerequisite before installing vLLM.
6. Restores NumPy 1.26.4 for compatibility with CANN/TBE and triton-ascend.
7. Replaces installed `vllm` and `vllm-ascend` with editable source installs,
   reusing the current environment instead of resolving a second isolated build
   environment.
8. Handles restricted-container network-interface inspection used by CANN/TBE.
9. Prints the installed versions and checked-out revisions.

Run it from any directory:

```bash
bash .devcontainer/skills/setup-vllm-ascend-dev/scripts/install-vllm-ascend.sh
```

Set `VLLM_DEV_ROOT` to override the default `$HOME/vllm-workspace`. Set
`VLLM_ASCEND_REF` to test a branch other than `main`. Set
`VLLM_SETUP_DRY_RUN=1` to validate resolution and patches without uninstalling
or installing packages.

Do not replace the verified vLLM commit with a guessed tag. Do not stash or
discard user changes in an existing checkout. The script stops when either
checkout is dirty, so inspect and preserve those changes before retrying.
