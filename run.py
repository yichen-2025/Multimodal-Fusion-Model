"""命令运行包装器。

用法：python run.py <真实命令>
示例：python run.py python scripts/train.py --variant A0* --epochs 3

职责：
1. 构造 CommandRecord（运行前填充通用属性 + 注册表差异化属性）
2. 执行真实命令（stdout 直接透传，stderr 透传并缓存）
3. 补全记录（duration / status / exit_code / error_message）
4. 追加 JSONL 到 run_history/YYYY-MM-DD.jsonl
5. 返回真实命令的退出码
"""
import sys
import os
import json
import subprocess
from datetime import datetime

# 确保能 import utils 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.command_record import CommandRecord
from utils.command_registry import COMMAND_REGISTRY, DEFAULT_COMMAND_META


def _extract_script(cmd_parts: list) -> str:
    """从命令行参数中提取脚本名。

    例如 ['python', 'scripts/train.py', '--variant', 'A0*'] -> 'train'
    若找不到 scripts/xxx.py 模式，返回空串。
    """
    for part in cmd_parts:
        # 匹配 scripts/xxx.py 或 scripts\xxx.py
        if "scripts" in part and part.endswith(".py"):
            # 取最后一级文件名，去掉 .py
            basename = os.path.basename(part)
            return basename[:-3]
    return ""


def _get_git_commit() -> str:
    """获取当前 HEAD commit short hash；失败返回空串。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def _extract_runtime_error(stderr_text: str) -> str:
    """从 stderr 提取运行时错误信息；命令行参数错误返回空串。

    原则：error_message 只记录运行时错误（脚本执行中抛出的异常），
    命令行本身的参数错误（argparse 报错）不计入。
    """
    if "Traceback" not in stderr_text:
        return ""
    # 取 Traceback 最后一行（通常是异常类型: 消息）
    lines = stderr_text.strip().splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        if stripped and not line.startswith(" "):
            return stripped
    return ""


def _append_record(record: CommandRecord) -> None:
    """将记录追加到 run_history/YYYY-MM-DD.jsonl。"""
    run_history_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "run_history"
    )
    os.makedirs(run_history_dir, exist_ok=True)

    date_str = record.execute_date
    file_path = os.path.join(run_history_dir, f"{date_str}.jsonl")

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def main():
    if len(sys.argv) < 2:
        print("用法：python run.py <真实命令>", file=sys.stderr)
        print("示例：python run.py python scripts/train.py --variant A0*", file=sys.stderr)
        sys.exit(1)

    real_cmd = sys.argv[1:]
    command_str = " ".join(real_cmd)
    script = _extract_script(real_cmd)

    meta = COMMAND_REGISTRY.get(script, DEFAULT_COMMAND_META)

    now = datetime.now()
    record = CommandRecord(
        command=command_str,
        execute_date=now.strftime("%Y-%m-%d"),
        duration_seconds=0.0,
        status="success",
        generated_files=list(meta["generated_files"]),
        modified_files=list(meta["modified_files"]),
        error_message="",
        notes=meta["notes"],
        timestamp=now.isoformat(),
        script=script,
        git_commit=_get_git_commit(),
    )

    # 执行命令：stdout 直接透传，stderr 透传并缓存
    start = datetime.now()
    try:
        proc = subprocess.Popen(
            real_cmd,
            stdout=sys.stdout,
            stderr=subprocess.PIPE,
        )
        stderr_lines = []
        # 逐行读取 stderr，透传到终端并缓存
        for line in proc.stderr:
            sys.stderr.write(line.decode("utf-8", errors="replace"))
            sys.stderr.flush()
            stderr_lines.append(line.decode("utf-8", errors="replace"))
        proc.wait()
    except FileNotFoundError:
        # 命令不存在
        end = datetime.now()
        record.duration_seconds = (end - start).total_seconds()
        record.exit_code = 127
        record.status = "failed"
        record.error_message = f"Command not found: {real_cmd[0] if real_cmd else ''}"
        _append_record(record)
        sys.exit(127)
    except Exception as e:
        # 其他启动异常
        end = datetime.now()
        record.duration_seconds = (end - start).total_seconds()
        record.exit_code = 1
        record.status = "failed"
        record.error_message = f"{type(e).__name__}: {e}"
        _append_record(record)
        sys.exit(1)

    end = datetime.now()
    record.duration_seconds = (end - start).total_seconds()
    record.exit_code = proc.returncode
    record.status = "success" if proc.returncode == 0 else "failed"
    record.error_message = _extract_runtime_error("".join(stderr_lines))

    _append_record(record)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
