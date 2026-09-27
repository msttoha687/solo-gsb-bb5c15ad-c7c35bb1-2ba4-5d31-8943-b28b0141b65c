"""样本直接读数与共享记录仪读数的温度一致性配对（纯函数，不依赖数据库与 Web）。

给定两侧已按事件时间排序的读数序列，在**不交叉一对一**（若第 ``i`` 条直接读数
与第 ``j`` 条记录仪读数配对，则后续配对只能取 ``i'>i, j'>j``）且配对两侧事件
时间差不超过 ``max_time_diff_seconds`` 的前提下，按以下字典序目标择优：

1. 配对数最多；
2. 所有配对的绝对时间差之和最小（时间差以微秒整数精确累加，避免浮点求和歧义）；
3. 仍相同时，取配对读数 ID 序列（``(直接读数 id, 记录仪读数 id)`` 元组序列）
   字典序最小者，保证结果稳定可复现。

动态规划：``f(i, j)`` 表示直接读数后缀 ``[i:]`` 与记录仪读数后缀 ``[j:]`` 的最优
配对；候选动作为跳过直接读数、跳过记录仪读数、配对当前两侧（仅当时间差在范围内）。
计数/时间差/首对 ID 三张紧凑表支持 O(1) 回溯；时间差窗口内的候选边用双指针划定。
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from datetime import datetime, timezone

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class MatchPoint:
    """参与配对的一侧读数。``id`` 为对应读数表的稳定主键。"""

    id: int
    event_time: str
    temperature_c: float


@dataclass(frozen=True)
class ConsistencyPair:
    """一条一对一配对（两侧读数与时间差/温差），按事件时间顺序产出。"""

    direct_reading_id: int
    direct_event_time: str
    direct_temperature_c: float
    logger_reading_id: int
    logger_event_time: str
    logger_temperature_c: float
    # 两侧事件时间的绝对差（秒，保留 6 位小数）
    time_diff_seconds: float
    # 两侧温差绝对值（°C，保留 6 位小数）；是否越界由调用方与阈值比较
    temperature_diff_c: float


@dataclass
class ConsistencyMatch:
    pairs: list[ConsistencyPair]
    direct_unpaired_count: int
    logger_unpaired_count: int
    # 全部配对的绝对时间差总和（秒）——第 2 优化目标的达成值，便于核查解释
    total_abs_time_diff_seconds: float

    @property
    def pair_count(self) -> int:
        return len(self.pairs)


def _parse_utc(event_time: str) -> datetime:
    """解析 UTC ISO 8601 时间（与 services.to_utc_iso 的输出一致）。"""

    dt = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _epoch_micros(dt: datetime) -> int:
    """相对于 Unix 纪元的整数微秒（datetime 减法精确到微秒，不走 float 转换）。"""

    delta = dt - _EPOCH
    return delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds


def match_temperatures(
    direct: list[MatchPoint],
    logger: list[MatchPoint],
    max_time_diff_seconds: float,
) -> ConsistencyMatch:
    """按“配对数最多 → 总时间差最小 → 配对 ID 序列字典序最小”求最优不交叉配对。

    两侧输入允许乱序，函数内部按 ``(event_time, id)`` 稳定排序后计算。
    """

    if max_time_diff_seconds < 0:
        raise ValueError("max_time_diff_seconds 不能为负")

    d_side = sorted(direct, key=lambda p: (p.event_time, p.id))
    l_side = sorted(logger, key=lambda p: (p.event_time, p.id))
    n, m = len(d_side), len(l_side)
    if n == 0 or m == 0:
        return ConsistencyMatch(
            pairs=[],
            direct_unpaired_count=n,
            logger_unpaired_count=m,
            total_abs_time_diff_seconds=0.0,
        )

    d_us = [_epoch_micros(_parse_utc(p.event_time)) for p in d_side]
    l_us = [_epoch_micros(_parse_utc(p.event_time)) for p in l_side]
    window_us = int(round(max_time_diff_seconds * 1_000_000))

    # 每个直接读数可配对的记录仪读数下标区间（时间差窗口内为连续区间）
    lo = [0] * n
    hi = [-1] * n
    j_lo = 0
    j_hi = -1
    for i in range(n):
        earliest = d_us[i] - window_us
        latest = d_us[i] + window_us
        while j_lo < m and l_us[j_lo] < earliest:
            j_lo += 1
        while j_hi + 1 < m and l_us[j_hi + 1] <= latest:
            j_hi += 1
        lo[i] = j_lo
        hi[i] = j_hi

    # (did, lid) 编码为单一整数：did 为主序、lid 为次序，字典序即编码大小
    id_base = max(p.id for p in l_side) + 1

    def _encode(direct_id: int, logger_id: int) -> int:
        return direct_id * id_base + logger_id

    width = m + 1
    size = (n + 1) * width
    # 三张紧凑表：配对数（越大越好）、时间差微秒总和（越小越好）、
    # 最优解首对读数 ID 编码（越小越好）。底行/右列全 0（空后缀）。
    counts = array("i", [0]) * size
    sums = array("q", [0]) * size
    firsts = array("q", [0]) * size

    def _better(c: int, s: int, f: int, best_c: int, best_s: int, best_f: int) -> bool:
        if c != best_c:
            return c > best_c
        if s != best_s:
            return s < best_s
        return f < best_f

    for i in range(n - 1, -1, -1):
        row = i * width
        next_row = (i + 1) * width
        for j in range(m - 1, -1, -1):
            idx = row + j

            # 候选 1：跳过当前直接读数
            k = next_row + j
            best_c, best_s, best_f = counts[k], sums[k], firsts[k]

            # 候选 2：跳过当前记录仪读数
            k = idx + 1
            c2, s2, f2 = counts[k], sums[k], firsts[k]
            if _better(c2, s2, f2, best_c, best_s, best_f):
                best_c, best_s, best_f = c2, s2, f2

            # 候选 3：配对两侧（仅当时间差在窗口内）
            if lo[i] <= j <= hi[i]:
                k = next_row + j + 1
                diff_us = abs(d_us[i] - l_us[j])
                c3 = counts[k] + 1
                s3 = sums[k] + diff_us
                f3 = _encode(d_side[i].id, l_side[j].id)
                if _better(c3, s3, f3, best_c, best_s, best_f):
                    best_c, best_s, best_f = c3, s3, f3

            counts[idx] = best_c
            sums[idx] = best_s
            firsts[idx] = best_f

    # 回溯：每个格点的最优值唯一决定一个候选（首对编码互不相同）
    matched_idx: list[tuple[int, int]] = []
    i = j = 0
    while True:
        idx = i * width + j
        target_c, target_s, target_f = counts[idx], sums[idx], firsts[idx]
        if target_c == 0:
            break

        chosen = None

        # 配对候选
        if lo[i] <= j <= hi[i]:
            k = (i + 1) * width + j + 1
            diff_us = abs(d_us[i] - l_us[j])
            c3 = counts[k] + 1
            s3 = sums[k] + diff_us
            f3 = _encode(d_side[i].id, l_side[j].id)
            if c3 == target_c and s3 == target_s and f3 == target_f:
                chosen = ("pair", diff_us)

        # 跳过直接读数
        if chosen is None:
            k = (i + 1) * width + j
            if (
                counts[k] == target_c
                and sums[k] == target_s
                and firsts[k] == target_f
            ):
                chosen = ("skip_direct", 0)

        # 跳过记录仪读数
        if chosen is None:
            k = i * width + j + 1
            if (
                counts[k] == target_c
                and sums[k] == target_s
                and firsts[k] == target_f
            ):
                chosen = ("skip_logger", 0)

        if chosen[0] == "pair":
            matched_idx.append((i, j, chosen[1]))
            i += 1
            j += 1
        elif chosen[0] == "skip_direct":
            i += 1
        else:
            j += 1

    pairs: list[ConsistencyPair] = []
    total_us = 0
    for i, j, diff_us in matched_idx:
        d, l = d_side[i], l_side[j]
        total_us += diff_us
        pairs.append(
            ConsistencyPair(
                direct_reading_id=d.id,
                direct_event_time=d.event_time,
                direct_temperature_c=d.temperature_c,
                logger_reading_id=l.id,
                logger_event_time=l.event_time,
                logger_temperature_c=l.temperature_c,
                time_diff_seconds=round(diff_us / 1_000_000, 6),
                temperature_diff_c=round(
                    abs(d.temperature_c - l.temperature_c), 6
                ),
            )
        )

    return ConsistencyMatch(
        pairs=pairs,
        direct_unpaired_count=n - len(pairs),
        logger_unpaired_count=m - len(pairs),
        total_abs_time_diff_seconds=round(total_us / 1_000_000, 6),
    )
