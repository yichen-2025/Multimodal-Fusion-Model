"""
运行追踪核心模块：计时、记录、指标提取、jsonl 写入。
纯标准库，不依赖任何第三方包。
"""

import json
import os
import re
import sys
import shutil
import threading
import subprocess
from datetime import datetime

# ---------- 路径常量 ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "run_history")

# ---------- 指标正则表 ----------
# Layer 3 优先：显式 [FINAL_RESULT] 标记；Layer 2：通用正则
METRIC_PATTERNS = [
    # Layer 3：显式标记
    (r'\[FINAL_RESULT\]\s*(.*)', '_final_result_raw'),
    # Layer 2：OOD 实验核心指标
    (r'(?i)\bAUROC[^\d]*([\d.]+)', 'AUROC'),
    (r'(?i)\bu[ _-]?f1[^\d]*([\d.]+)', 'uF1'),
    (r'(?i)unknown[_ ]?f1[^\d]*([\d.]+)', 'unknown_f1'),
    (r'(?i)(?:macro[_ -]?f1|macro_f1)[^\d]*([\d.]+)', 'macro_f1'),
    (r'(?i)known[_ -]?accuracy[^\d]*([\d.]+)', 'known_accuracy'),
    (r'(?i)(?:unknown[_ -]?leak|leak(?:age)?[_ -]?rate)[^\d]*([\d.]+)', 'unknown_leak_rate'),
    # Layer 2：通用分类指标
    (r'(?i)precision[^\d]*([\d.]+)', 'precision'),
    (r'(?i)(?:unknown[_ -]?recall|recall)[^\d]*([\d.]+)', 'recall'),
    (r'(?i)(?<!macro[_ -])(?<!unknown[_ -])\bf1[^\d]*([\d.]+)', 'f1'),
    (r'(?i)(?:整体)?accuracy[^\d]*([\d.]+)', 'accuracy'),
    (r'(?i)\bloss[^\d]*([\d.]+)', 'loss'),
    # 验证脚本专用
    (r'(?i)FAIL[^\d]*(\d+)', 'fail_count'),
    (r'(?i)WARN[^\d]*(\d+)', 'warn_count'),
]

# ---------- 线程锁 ----------
_write_lock = threading.Lock()


# ============================================================
#  工具函数区
# ============================================================

def get_git_info():
    """获取当前 git 分支和 commit hash，失败返回空 dict"""
    info = {}
    try:
        branch = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=5,
            encoding='utf-8', errors='replace'
        )
        if branch.returncode == 0:
            info['git_branch'] = branch.stdout.strip()
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=5,
            encoding='utf-8', errors='replace'
        )
        if commit.returncode == 0:
            info['git_commit'] = commit.stdout.strip()[:7]
    except Exception:
        pass
    return info


def humanize_duration(seconds):
    """秒数 → 可读字符串：1842.73 → '30m42s'；0.19 → '0.19s'"""
    if seconds < 1:
        return f"{seconds:.2f}s"
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m}m{s}s"
    elif m > 0:
        return f"{m}m{s}s"
    else:
        return f"{s}s"


def extract_error_message(stderr_text):
    """
    从 stderr 提取错误摘要。
    策略：从后往前找第一个包含 Error/Exception/Traceback 的行，
    找不到就取最后 5 行。
    """
    lines = stderr_text.strip().splitlines()
    if not lines:
        return ""
    error_keywords = re.compile(
        r'(Error|Exception|Traceback|ModuleNotFoundError|'
        r'ImportError|FileNotFoundError|OSError|RuntimeError|'
        r'AttributeError|TypeError|ValueError|KeyError|IndexError)'
    )
    for line in reversed(lines):
        if error_keywords.search(line):
            return line.strip()
    return '\n'.join(lines[-5:])


def extract_metrics(stdout_text):
    """
    从 stdout 提取指标。
    优先 Layer 3（[FINAL_RESULT]），没命中再跑 Layer 2（通用正则）。
    """
    metrics = {}

    # Layer 3：显式标记
    final_match = re.search(r'\[FINAL_RESULT\]\s*(.*)', stdout_text)
    if final_match:
        raw = final_match.group(1)
        for kv in raw.split():
            if '=' in kv:
                k, v = kv.split('=', 1)
                try:
                    metrics[k.strip()] = float(v.strip())
                except ValueError:
                    metrics[k.strip()] = v.strip()
        if metrics:
            return metrics

    # Layer 2：通用正则
    for pattern, key in METRIC_PATTERNS:
        if key == '_final_result_raw':
            continue
        match = re.search(pattern, stdout_text)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except ValueError:
                metrics[key] = match.group(1)

    return metrics


# ============================================================
#  核心写入函数
# ============================================================

def record_run(entry):
    """
    追加一条运行记录到当日 jsonl 文件。
    线程安全 + flush + fsync + 磁盘/大小检查。
    """
    try:
        os.makedirs(LOG_DIR, exist_ok=True)

        # 磁盘空间检查（低于 100MB 警告）
        try:
            free = shutil.disk_usage(LOG_DIR).free
            if free < 100 * 1024 * 1024:
                print("[run_tracker] 警告：磁盘剩余不足 100MB", file=sys.stderr)
        except Exception:
            pass

        today = datetime.now().strftime('%Y-%m-%d')
        filepath = os.path.join(LOG_DIR, f"{today}.jsonl")

        # 单文件大小检查（超过 50MB 警告）
        try:
            if os.path.exists(filepath) and os.path.getsize(filepath) > 50 * 1024 * 1024:
                print(f"[run_tracker] 警告：今日日志已超 50MB ({filepath})", file=sys.stderr)
        except Exception:
            pass

        line = json.dumps(entry, ensure_ascii=False) + '\n'

        with _write_lock:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(line)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass  # Windows 下 fsync 行为不完全一致，忽略

    except Exception as e:
        print(f"[run_tracker] 写入失败：{e}", file=sys.stderr)
