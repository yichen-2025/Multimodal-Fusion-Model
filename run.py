"""
运行命令追踪 Wrapper — 零侵入执行 + 自动记录。

用法：
    python run.py scripts/train.py --lr 1e-4 --seed 23        # 正常执行 + 记录
    python run.py --no-check scripts/train.py ...             # 跳过语法预检

工作原理：
    1. 预检（文件存在性 + 可选语法检查）——失败直接退出，不记录
    2. subprocess.Popen 执行真实命令，stdout/stderr 实时透传
    3. 收集完整 stdout/stderr，尾部截断 + 指标正则提取
    4. 追加一行 JSON 到 run_history/YYYY-MM-DD.jsonl
    5. sys.exit(exit_code) 返回与真实命令一致的退出码
"""

import os
import sys
import re
import time
import uuid
import threading
import subprocess
from datetime import datetime

# ---------- 尝试加载 run_tracker（失败则降级：命令照常执行，但跳过追踪） ----------
try:
    from utils.run_tracker import (
        record_run, get_git_info, humanize_duration,
        extract_error_message, extract_metrics
    )
    TRACKER_AVAILABLE = True
except ImportError as e:
    print(f"[run.py] 警告：run_tracker 加载失败 ({e})，跳过追踪", file=sys.stderr)
    TRACKER_AVAILABLE = False


# ============================================================
#  管道实时读取（同时透传到终端 + 收集全文）
# ============================================================

def stream_reader(pipe, collector, target_pipe):
    """
    逐行读取子进程管道：
    - 追加到 collector 列表（用于事后截断/提取）
    - 同时 write 到 target_pipe（终端实时输出，用户体验不降级）
    """
    for line in pipe:
        collector.append(line)
        target_pipe.write(line)
        target_pipe.flush()
    pipe.close()


# ============================================================
#  核心执行 + 追踪逻辑
# ============================================================

def run_command(args, skip_check=False):
    """
    执行 python 命令并记录。

    Args:
        args: list[str] — 传给 python 的完整参数，如 ['scripts/train.py', '--lr', '1e-4']
        skip_check: bool — 是否跳过文件/语法预检

    Returns:
        int — 真实进程的退出码
    """

    # ---------- 阶段 1：预检（失败直接 return，不记录） ----------
    if not skip_check and args and args[0].endswith('.py'):
        script_path = args[0]
        if not os.path.isfile(script_path):
            print(f"[run.py] 错误：文件不存在 → {script_path}", file=sys.stderr)
            return 1
        # 语法快速检查
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'py_compile', script_path],
                capture_output=True, text=True, timeout=10,
                encoding='utf-8', errors='replace'
            )
            if result.returncode != 0:
                print(f"[run.py] 错误：语法检查失败", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                return 1
        except Exception:
            pass  # py_compile 超时或异常不阻断，继续执行

    # ---------- 阶段 2：准备 ----------
    start_time = time.time()
    git_info = get_git_info() if TRACKER_AVAILABLE else {}

    # ---------- 阶段 3：执行 subprocess ----------
    stdout_lines = []
    stderr_lines = []
    status = 'success'
    exit_code = 0
    error_msg = ''

    try:
        proc = subprocess.Popen(
            [sys.executable] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=os.getcwd()
        )

        t_out = threading.Thread(target=stream_reader, args=(proc.stdout, stdout_lines, sys.stdout))
        t_err = threading.Thread(target=stream_reader, args=(proc.stderr, stderr_lines, sys.stderr))
        t_out.daemon = True
        t_err.daemon = True
        t_out.start()
        t_err.start()

        exit_code = proc.wait()
        t_out.join()
        t_err.join()

        # 根据退出码 + stderr 内容分类 status
        if exit_code != 0:
            stderr_text = ''.join(stderr_lines)
            if re.search(r'(SyntaxError|argparse|ModuleNotFoundError|ImportError)', stderr_text):
                status = 'error'       # 环境/语法问题
            else:
                status = 'failed'      # 运行时业务失败
            error_msg = extract_error_message(stderr_text)

    except KeyboardInterrupt:
        status = 'interrupted'
        exit_code = -1
        print("\n[run.py] 进程被用户中断", file=sys.stderr)

    except OSError as e:
        status = 'system_error'
        exit_code = -1
        error_msg = str(e)
        print(f"[run.py] 系统错误：{e}", file=sys.stderr)

    except Exception as e:
        status = 'error'
        exit_code = -1
        error_msg = str(e)
        print(f"[run.py] 未知错误：{e}", file=sys.stderr)

    # ---------- 阶段 4：记录 ----------
    if TRACKER_AVAILABLE:
        try:
            duration = time.time() - start_time
            stdout_text = ''.join(stdout_lines)
            stderr_text = ''.join(stderr_lines)

            # Layer 1：尾部截断
            tail_stdout = '\n'.join(stdout_lines[-50:]).strip()
            tail_stderr = '\n'.join(stderr_lines[-20:]).strip()

            # Layer 2 + Layer 3：指标提取（仅成功/失败/中断时尝试）
            metrics = extract_metrics(stdout_text) if status in ('success', 'failed', 'interrupted') else {}

            entry = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': round(duration, 2),
                'duration_human': humanize_duration(duration),
                'command': f"python {' '.join(args)}",
                'working_dir': os.getcwd(),
                'exit_code': exit_code,
                'status': status,
                'error_message': error_msg if error_msg else None,
                'extracted_metrics': metrics if metrics else None,
                'stdout_tail': tail_stdout,
                'stderr_tail': tail_stderr if tail_stderr else None,
            }
            entry.update(git_info)

            record_run(entry)
            print(
                f"\n[run.py] ✓ 已记录  状态={status}  "
                f"耗时={humanize_duration(duration)}  "
                f"退出码={exit_code}",
                file=sys.stderr
            )
        except Exception as e:
            print(f"[run.py] 记录写入失败（不影响命令本身）：{e}", file=sys.stderr)

    return exit_code


# ============================================================
#  入口
# ============================================================

def main():
    args = sys.argv[1:]

    # 分离 wrapper 自有参数和真实命令参数
    skip_check = '--no-check' in args
    if skip_check:
        args.remove('--no-check')

    if not args:
        print(
            "用法：python run.py [--no-check] <python_script> [script_args...]\n"
            "示例：python run.py scripts/train.py --lr 1e-4 --seed 23",
            file=sys.stderr
        )
        sys.exit(1)

    exit_code = run_command(args, skip_check=skip_check)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
