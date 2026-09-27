"""温度一致性配对纯函数单元测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.matching import ConsistencyPair, MatchPoint, match_temperatures


_BASE = datetime(2026, 9, 20, 8, 0, 0, tzinfo=timezone.utc)


def _p(id: int, minute: float, temp: float) -> MatchPoint:
    return MatchPoint(
        id=id,
        event_time=(_BASE + timedelta(minutes=minute)).isoformat(),
        temperature_c=temp,
    )


def _ids(result) -> list[tuple[int, int]]:
    return [(p.direct_reading_id, p.logger_reading_id) for p in result.pairs]


def test_empty_sides() -> None:
    r = match_temperatures([], [], 60)
    assert r.pairs == []
    assert r.pair_count == 0
    assert r.direct_unpaired_count == 0
    assert r.logger_unpaired_count == 0

    one_side = match_temperatures([_p(1, 0, 4.0)], [], 60)
    assert one_side.pair_count == 0
    assert one_side.direct_unpaired_count == 1
    assert one_side.logger_unpaired_count == 0


def test_negative_window_rejected() -> None:
    with pytest.raises(ValueError):
        match_temperatures([_p(1, 0, 1)], [_p(2, 0, 1)], -1)


def test_max_count_non_crossing_basic() -> None:
    direct = [_p(1, 0, 4), _p(2, 10, 4), _p(3, 20, 4)]
    logger = [_p(10, 0, 4), _p(20, 20, 4)]
    r = match_temperatures(direct, logger, max_time_diff_seconds=0)
    # 窗口为 0：仅同时刻可配对，中间的直接读数未配对
    assert _ids(r) == [(1, 10), (3, 20)]
    assert r.direct_unpaired_count == 1
    assert r.logger_unpaired_count == 0


def test_window_boundary_inclusive() -> None:
    # 时间差恰好 60 秒：取等号仍可配对
    r = match_temperatures(
        [_p(1, 0, 4)], [_p(2, 1, 4)], max_time_diff_seconds=60
    )
    assert r.pair_count == 1
    assert r.pairs[0].time_diff_seconds == 60.0

    # 超出一秒不可配对
    r = match_temperatures(
        [_p(1, 0, 4)], [_p(2, 1 + 1 / 60, 4)], max_time_diff_seconds=60
    )
    assert r.pair_count == 0
    assert r.direct_unpaired_count == 1
    assert r.logger_unpaired_count == 1


def test_min_total_time_diff_among_max_cardinality() -> None:
    # 直接：0 / 100 分钟；记录仪：0 / 60 / 100 分钟；窗口 100 分钟
    # 两种 2 配对方案：(0,0)+(100,60)=40min 与 (0,0)+(100,100)=0min
    # 必须取总时间差最小的后者
    direct = [_p(1, 0, 4), _p(2, 100, 4)]
    logger = [_p(10, 0, 4), _p(20, 60, 4), _p(30, 100, 4)]
    r = match_temperatures(direct, logger, max_time_diff_seconds=100 * 60)
    assert _ids(r) == [(1, 10), (2, 30)]
    assert r.total_abs_time_diff_seconds == 0.0


def test_lex_smallest_id_sequence_tiebreak() -> None:
    # 三对同时刻直接读数、两条同时刻记录仪读数：只能选两条直接读数，
    # 方案 [(1,10),(2,20)] / [(1,10),(3,20)] / [(2,10),(3,20)] 同数同总差，
    # 取配对读数 ID 序列字典序最小者
    direct = [_p(1, 0, 4), _p(2, 0, 4), _p(3, 0, 4)]
    logger = [_p(10, 0, 4), _p(20, 0, 4)]
    r = match_temperatures(direct, logger, max_time_diff_seconds=0)
    assert _ids(r) == [(1, 10), (2, 20)]
    assert r.direct_unpaired_count == 1
    assert r.logger_unpaired_count == 0


def test_deterministic_under_reversed_input() -> None:
    # 输入乱序不影响结果（内部按 (event_time, id) 排序）
    direct = [_p(3, 20, 4), _p(1, 0, 4), _p(2, 10, 4)]
    logger = [_p(20, 20, 4), _p(10, 0, 4)]
    r = match_temperatures(direct, logger, max_time_diff_seconds=0)
    assert _ids(r) == [(1, 10), (3, 20)]
    # 配对按事件时间顺序返回
    times = [p.direct_event_time for p in r.pairs]
    assert times == sorted(times)


def test_temperature_diff_fields() -> None:
    r = match_temperatures(
        [_p(1, 0, 4.0)],
        [_p(2, 0.5, -1.5)],
        max_time_diff_seconds=60,
    )
    assert r.pair_count == 1
    pair: ConsistencyPair = r.pairs[0]
    assert pair.temperature_diff_c == 5.5
    assert pair.time_diff_seconds == 30.0
    assert pair.direct_temperature_c == 4.0
    assert pair.logger_temperature_c == -1.5


def test_microsecond_time_diff_exact() -> None:
    # 0.5 秒的时间差精确累加与呈现（不依赖浮点求和误差）
    r = match_temperatures(
        [_p(1, 0, 1.0)],
        [_p(2, 0.5 / 60, 1.0)],
        max_time_diff_seconds=1,
    )
    assert r.pairs[0].time_diff_seconds == 0.5
    assert r.total_abs_time_diff_seconds == 0.5


def test_matches_bruteforce_optimum() -> None:
    """与小规模全枚举对照：配对数、总时间差必须一致；ID 序列须字典序最小。"""

    import itertools

    direct = [_p(i, i * 7 + (i % 3), float(i)) for i in range(1, 6)]
    logger = [_p(100 + j, j * 9 + (j % 2), float(j) + 0.4) for j in range(1, 5)]
    window = 20 * 60

    r = match_temperatures(direct, logger, max_time_diff_seconds=window)

    def _us(p: MatchPoint) -> int:
        dt = datetime.fromisoformat(p.event_time)
        delta = dt - _BASE
        return delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds

    d_us = [_us(p) for p in direct]
    l_us = [_us(p) for p in logger]
    win_us = int(window * 1_000_000)

    best = None  # (count, -sum 最大化, 序列)
    for k in range(min(len(direct), len(logger)), 0, -1):
        found = []
        for di in itertools.combinations(range(len(direct)), k):
            for li in itertools.combinations(range(len(logger)), k):
                pairs = list(zip(di, li))
                if all(abs(d_us[a] - l_us[b]) <= win_us for a, b in pairs):
                    total = sum(abs(d_us[a] - l_us[b]) for a, b in pairs)
                    seq = [(direct[a].id, logger[b].id) for a, b in pairs]
                    found.append((total, seq))
        if found:
            min_sum = min(s for s, _ in found)
            seq = min(q for s, q in found if s == min_sum)
            best = (k, min_sum, seq)
            break

    assert best is not None
    count, total_us, seq = best
    assert r.pair_count == count
    assert int(round(r.total_abs_time_diff_seconds * 1_000_000)) == total_us
    assert _ids(r) == seq
