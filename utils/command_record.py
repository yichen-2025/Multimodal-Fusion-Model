"""
命令运行记录对象。

把每次命令行运行抽象为一个 CommandRecord 对象，
属性明确、可扩展，替代已删除的 run_tracker 正则提取方案。
"""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class CommandRecord:
    """
    一次命令行运行的完整记录
    """

    # -- 用户指定的核心属性 --
    command: str                       # 命令行内容（完整）
    execute_date: str                  # 执行日期，格式 YYYY-MM-DD
    duration_seconds: float            # 运行时间（秒）
    status: str                        # 执行状态：success / failed
    generated_files: list              # 生成的文件（说明性文本列表，无则空列表）
    modified_files: list               # 改动的文件（说明性文本列表，无则空列表）
    error_message: str                 # 报错说明（仅运行时错误；命令行本身错误为空串）
    notes: str                         # 附加说明（来自注册表 + 可运行时追加）

    # -- 辅助属性（自动填充，便于追溯） --
    execute_time: str = ""              # 执行时间，格式 HH:MM:SS
    timestamp: str = ""                # 完整时间戳 ISO 格式
    exit_code: Optional[int] = None    # 真实命令退出码
    script: str = ""                   # 脚本名（从 command 解析，用于查注册表）
    git_commit: str = ""               # 当前 HEAD commit short hash

    # -- 扩展位：后续新增属性放这里，避免改动已有字段结构 --
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        序列化为 dict（用于写 JSONL）
        """
        return asdict(self)
