# Restore Notes for VM Setup

## 2025-08-28: Results from merged setup.sh

### 1. Hardlink Warnings
- **Details:** Multiple warnings: `Failed to hardlink files; falling back to full copy. This may lead to degraded performance.`
- **Action Needed:** This is usually safe, but if performance is a concern, set `export UV_LINK_MODE=copy` or use `--link-mode=copy` in the script to suppress the warning.

### 2. CUDA not available
- **Details:** The environment validation step reports `CUDA available: False` and exits with code 1.
- **Action Needed:**
  - Check if the VM has a GPU and that NVIDIA drivers are installed and loaded.
  - If this is a CPU-only VM, update the setup script to handle this gracefully.

### 3. General Success
- **Details:** All other steps (system packages, Python, dependencies, project install) completed successfully.

---
