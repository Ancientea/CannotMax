import pandas as pd
from math import sqrt
from collections import defaultdict
from src.core.config import MONSTER_COUNT, FIELD_FEATURE_COUNT, MONSTER_DATA
from src.core.paths import image_path, PROJECT_ROOT

FIELD_FEATURE_COUNT = 0


def load_data():
    """加载数据"""
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "arknights.csv", header=None, low_memory=False
    )
    # 设置列名
    monster_cols_left = [f"L{i+1}" for i in range(MONSTER_COUNT)]
    field_cols_left = [f"FL{i+1}" for i in range(FIELD_FEATURE_COUNT)]
    monster_cols_right = [f"R{i+1}" for i in range(MONSTER_COUNT)]
    field_cols_right = [f"FR{i+1}" for i in range(FIELD_FEATURE_COUNT)]

    df.columns = (
        monster_cols_left
        + field_cols_left
        + monster_cols_right
        + field_cols_right
        + ["Result", "ImgPath"]
    )
    return df


def get_monster_name(monster_id):
    """根据怪物ID获取怪物名称"""
    if monster_id in MONSTER_DATA.index:
        return MONSTER_DATA.loc[monster_id]["名称"]
    return f"怪物{monster_id}"


def get_monster_original_name(monster_id):
    """根据怪物ID获取怪物原始名称（用于匹配图片）"""
    if monster_id in MONSTER_DATA.index:
        return MONSTER_DATA.loc[monster_id]["原始名称"]
    return f"怪物{monster_id}"


def monster_image_uri(monster_name):
    path = image_path(monster_name)
    if not path.exists():
        path = image_path("empty")
    return path.resolve().as_uri()


def monster_img_tag(monster_name, size):
    uri = monster_image_uri(monster_name)
    empty_uri = monster_image_uri("empty")
    return (
        f'<img src="{uri}" onerror="this.src=\'{empty_uri}\'" '
        f'style="width:{size}px;height:{size}px;">'
    )


def calculate_all_monster_win_rates(df):
    """计算所有怪物的胜率"""
    monster_stats = {}
    total_matches = len(df)

    for i in range(1, MONSTER_COUNT + 1):
        try:
            # 转换数据类型并过滤
            df[f"L{i}"] = pd.to_numeric(df[f"L{i}"], errors="coerce").fillna(0)
            df[f"R{i}"] = pd.to_numeric(df[f"R{i}"], errors="coerce").fillna(0)

            # 左方统计
            left_games = df[df[f"L{i}"] != 0]
            left_wins = len(left_games[left_games["Result"] == "L"])
            left_total = len(left_games)

            # 右方统计
            right_games = df[df[f"R{i}"] != 0]
            right_wins = len(right_games[right_games["Result"] == "R"])
            right_total = len(right_games)
        except Exception as e:
            print(f"处理怪物{i}时出错: {e}")
            continue

        # 合并统计
        total_games = left_total + right_total
        total_wins = left_wins + right_wins

        if total_games > 0:
            monster_name = get_monster_name(i)
            win_rate = total_wins / total_games
            participation_rate = (
                total_games / total_matches if total_matches > 0 else 0
            )
            monster_stats[monster_name] = {
                "怪物ID": i,
                "胜场": total_wins,
                "总场数": total_games,
                "胜率": win_rate,
                "参战率": participation_rate,
            }

    return pd.DataFrame(monster_stats).T.sort_values("胜率", ascending=False)


def analyze_monster_combinations(df):
    """分析怪物配合效果"""
    # 初始化数据结构
    single_stats = defaultdict(lambda: {"appearances": 0, "wins": 0})
    pair_stats = defaultdict(lambda: {"co_occurrences": 0, "co_wins": 0})

    for _, record in df.iterrows():
        victory_side = record["Result"]

        # 获取左右两方的怪物
        left_monsters = []
        right_monsters = []

        for i in range(1, MONSTER_COUNT + 1):
            try:
                left_val = (
                    float(record[f"L{i}"]) if pd.notna(record[f"L{i}"]) else 0
                )
                right_val = (
                    float(record[f"R{i}"]) if pd.notna(record[f"R{i}"]) else 0
                )

                if left_val > 0:
                    left_monsters.append(i)
                if right_val > 0:
                    right_monsters.append(i)
            except (ValueError, TypeError):
                continue

        # 确定胜利队伍和失败队伍
        if victory_side == "L":
            win_team, lose_team = left_monsters, right_monsters
        else:
            win_team, lose_team = right_monsters, left_monsters

        # 更新单怪统计（胜利方）
        for monster in win_team:
            single_stats[monster]["appearances"] += 1
            single_stats[monster]["wins"] += 1

        # 更新单怪统计（失败方）
        for monster in lose_team:
            single_stats[monster]["appearances"] += 1

        # 更新双怪组合统计（胜利方）
        for i in range(len(win_team)):
            for j in range(i + 1, len(win_team)):
                x, y = sorted((win_team[i], win_team[j]))
                pair_stats[(x, y)]["co_occurrences"] += 1
                pair_stats[(x, y)]["co_wins"] += 1

        # 更新双怪组合统计（失败方）
        for i in range(len(lose_team)):
            for j in range(i + 1, len(lose_team)):
                x, y = sorted((lose_team[i], lose_team[j]))
                pair_stats[(x, y)]["co_occurrences"] += 1

    # 计算最佳配合
    results = []
    total_battles = len(df)

    for (x, y), stats in pair_stats.items():
        if stats["co_occurrences"] < 10:  # 过滤低频组合
            continue

        # 组合胜率
        xy_win_rate = stats["co_wins"] / stats["co_occurrences"]

        # 单怪胜率
        if (
            single_stats[x]["appearances"] > 0
            and single_stats[y]["appearances"] > 0
        ):
            x_win_rate = (
                single_stats[x]["wins"] / single_stats[x]["appearances"]
            )
            y_win_rate = (
                single_stats[y]["wins"] / single_stats[y]["appearances"]
            )

            # 提升度 - 简化计算，不使用卡方检验
            expected_win_rate = sqrt(x_win_rate * y_win_rate)
            lift = (
                xy_win_rate / expected_win_rate if expected_win_rate > 0 else 0
            )

            if (
                lift > 1.1
                and xy_win_rate > max(x_win_rate, y_win_rate)
                and stats["co_occurrences"] >= 20
            ):
                x_name = get_monster_name(x)
                y_name = get_monster_name(y)
                results.append(
                    {
                        "组合": f"{x_name}+{y_name}",
                        "怪物1": x_name,
                        "怪物2": y_name,
                        "ID1": x,
                        "ID2": y,
                        "提升度": lift,
                        "组合胜率": xy_win_rate,
                        "出场次数": stats["co_occurrences"],
                        "获胜次数": stats["co_wins"],
                    }
                )

    # 按提升度排序
    results.sort(key=lambda x: -x["提升度"])
    return pd.DataFrame(results)


def find_countered_monsters(df):
    """寻找被克制的怪物前五个"""
    counter_stats = defaultdict(lambda: {"total_matchups": 0, "losses": 0})

    for _, record in df.iterrows():
        victory_side = record["Result"]

        # 获取左右两方的怪物
        left_monsters = []
        right_monsters = []

        for i in range(1, MONSTER_COUNT + 1):
            try:
                left_val = (
                    float(record[f"L{i}"]) if pd.notna(record[f"L{i}"]) else 0
                )
                right_val = (
                    float(record[f"R{i}"]) if pd.notna(record[f"R{i}"]) else 0
                )

                if left_val > 0:
                    left_monsters.append(i)
                if right_val > 0:
                    right_monsters.append(i)
            except (ValueError, TypeError):
                continue

        # 分析对战情况
        for left_monster in left_monsters:
            for right_monster in right_monsters:
                # 左方怪物的统计
                counter_stats[left_monster]["total_matchups"] += 1
                if victory_side == "R":  # 左方败北
                    counter_stats[left_monster]["losses"] += 1

                # 右方怪物的统计
                counter_stats[right_monster]["total_matchups"] += 1
                if victory_side == "L":  # 右方败北
                    counter_stats[right_monster]["losses"] += 1

    # 计算被克制率
    countered_results = []
    for monster_id, stats in counter_stats.items():
        if stats["total_matchups"] >= 20:  # 至少20场对战
            loss_rate = stats["losses"] / stats["total_matchups"]
            monster_name = get_monster_name(monster_id)
            countered_results.append(
                {
                    "怪物": monster_name,
                    "怪物ID": monster_id,
                    "被克制率": loss_rate,
                    "败场": stats["losses"],
                    "总对战数": stats["total_matchups"],
                }
            )

    # 按被克制率排序（降序）
    countered_results.sort(key=lambda x: -x["被克制率"])
    return pd.DataFrame(countered_results[:5])


def analyze_individual_monster_relations(df):
    """分析每个怪物的详细关系：最佳队友、克制关系、被克制关系"""
    monster_relations = {}

    # 初始化数据结构
    single_stats = defaultdict(lambda: {"appearances": 0, "wins": 0})
    pair_stats = defaultdict(lambda: {"co_occurrences": 0, "co_wins": 0})
    counter_stats = defaultdict(
        lambda: defaultdict(lambda: {"matchups": 0, "wins": 0})
    )

    for _, record in df.iterrows():
        victory_side = record["Result"]

        # 获取左右两方的怪物
        left_monsters = []
        right_monsters = []

        for i in range(1, MONSTER_COUNT + 1):
            try:
                left_val = (
                    float(record[f"L{i}"]) if pd.notna(record[f"L{i}"]) else 0
                )
                right_val = (
                    float(record[f"R{i}"]) if pd.notna(record[f"R{i}"]) else 0
                )

                if left_val > 0:
                    left_monsters.append(i)
                if right_val > 0:
                    right_monsters.append(i)
            except (ValueError, TypeError):
                continue

        # 确定胜利队伍和失败队伍
        if victory_side == "L":
            win_team, lose_team = left_monsters, right_monsters
        else:
            win_team, lose_team = right_monsters, left_monsters

        # 更新单怪统计
        for monster in win_team:
            single_stats[monster]["appearances"] += 1
            single_stats[monster]["wins"] += 1

        for monster in lose_team:
            single_stats[monster]["appearances"] += 1

        # 更新队友统计
        for i in range(len(win_team)):
            for j in range(i + 1, len(win_team)):
                x, y = sorted((win_team[i], win_team[j]))
                pair_stats[(x, y)]["co_occurrences"] += 1
                pair_stats[(x, y)]["co_wins"] += 1

        for i in range(len(lose_team)):
            for j in range(i + 1, len(lose_team)):
                x, y = sorted((lose_team[i], lose_team[j]))
                pair_stats[(x, y)]["co_occurrences"] += 1

        # 更新克制关系统计
        for winner in win_team:
            for loser in lose_team:
                counter_stats[winner][loser]["matchups"] += 1
                counter_stats[winner][loser]["wins"] += 1
                counter_stats[loser][winner]["matchups"] += 1

    # 为每个怪物分析关系
    for monster_id in range(1, MONSTER_COUNT + 1):
        monster_name = get_monster_name(monster_id)

        if single_stats[monster_id]["appearances"] < 10:  # 数据量太少
            continue

        # 分析最佳队友
        best_teammates = []
        for (x, y), stats in pair_stats.items():
            if x == monster_id or y == monster_id:
                partner_id = y if x == monster_id else x
                if stats["co_occurrences"] >= 5:  # 至少5次合作
                    combo_win_rate = stats["co_wins"] / stats["co_occurrences"]

                    # 计算提升度
                    if (
                        single_stats[monster_id]["appearances"] > 0
                        and single_stats[partner_id]["appearances"] > 0
                    ):
                        monster_win_rate = (
                            single_stats[monster_id]["wins"]
                            / single_stats[monster_id]["appearances"]
                        )
                        partner_win_rate = (
                            single_stats[partner_id]["wins"]
                            / single_stats[partner_id]["appearances"]
                        )
                        expected_win_rate = sqrt(
                            monster_win_rate * partner_win_rate
                        )

                        if expected_win_rate > 0:
                            lift = combo_win_rate / expected_win_rate
                            if lift > 1.0:
                                best_teammates.append(
                                    {
                                        "partner_id": partner_id,
                                        "partner_name": get_monster_name(
                                            partner_id
                                        ),
                                        "lift": lift,
                                        "combo_win_rate": combo_win_rate,
                                        "occurrences": stats["co_occurrences"],
                                    }
                                )

        best_teammates.sort(key=lambda x: -x["lift"])

        # 分析克制关系
        counters = []  # 该怪物克制的
        countered_by = []  # 克制该怪物的

        for opponent_id in counter_stats[monster_id]:
            stats = counter_stats[monster_id][opponent_id]
            if stats["matchups"] >= 5:
                win_rate = stats["wins"] / stats["matchups"]
                if win_rate > 0.6:  # 胜率超过60%认为克制
                    counters.append(
                        {
                            "opponent_id": opponent_id,
                            "opponent_name": get_monster_name(opponent_id),
                            "win_rate": win_rate,
                            "matchups": stats["matchups"],
                        }
                    )

        for opponent_id in range(1, MONSTER_COUNT + 1):
            if (
                opponent_id in counter_stats
                and monster_id in counter_stats[opponent_id]
            ):
                stats = counter_stats[opponent_id][monster_id]
                if stats["matchups"] >= 5:
                    lose_rate = stats["wins"] / stats["matchups"]
                    if lose_rate > 0.6:  # 对方胜率超过60%认为被克制
                        countered_by.append(
                            {
                                "opponent_id": opponent_id,
                                "opponent_name": get_monster_name(opponent_id),
                                "lose_rate": lose_rate,
                                "matchups": stats["matchups"],
                            }
                        )

        counters.sort(key=lambda x: -x["win_rate"])
        countered_by.sort(key=lambda x: -x["lose_rate"])

        monster_relations[monster_id] = {
            "name": monster_name,
            "best_teammates": best_teammates[:3],
            "counters": counters[:3],
            "countered_by": countered_by[:3],
        }

    return monster_relations


def get_terrain_feature_columns():
    """获取地形特征列名"""
    import json
    import re
    from collections import defaultdict

    try:
        # 加载类别映射
        class_map_path = "tools/battlefield_recognize/class_to_idx.json"
        with open(class_map_path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)

        # 使用与data_cleaning_with_field_recognize_gpu.py相同的逻辑
        grouped_elements = defaultdict(list)
        for class_name in class_to_idx.keys():
            if class_name.endswith("_none"):
                continue
            condensed_name = re.sub(r"_left_", "_", class_name)
            condensed_name = re.sub(r"_right_", "_", condensed_name)
            grouped_elements[condensed_name].append(class_name)

        # 返回排序后的特征列名
        return sorted(grouped_elements.keys())
    except Exception as e:
        print(f"无法获取地形特征列名，使用默认值: {e}")
        # 如果无法获取，返回默认列表
        return [
            "altar_vertical_altar",
            "block_parallel_block",
            "block_vertical_altar_shape1",
            "block_vertical_altar_shape2",
            "block_vertical_block_shape1",
            "block_vertical_block_shape2",
            "coil_narrow_coil",
            "coil_wide_coil",
            "crossbow_top_crossbow",
            "fire_side_crossbow",
            "fire_side_fire",
            "fire_top_fire",
        ]


def analyze_terrain_effects(df):
    """分析地形对怪物的影响"""
    terrain_effects = []

    # 获取实际的地形特征列名
    terrain_feature_columns = get_terrain_feature_columns()

    # 地形显示名称映射
    terrain_display_mapping = {
        "altar_vertical_altar": "垂直祭坛",
        "block_parallel_block": "平行方块阻挡",
        "block_vertical_altar_shape1": "垂直祭坛形阻挡1",
        "block_vertical_altar_shape2": "垂直祭坛形阻挡2",
        "block_vertical_block_shape1": "垂直方块阻挡1",
        "block_vertical_block_shape2": "垂直方块阻挡2",
        "coil_narrow_coil": "窄型线圈装置",
        "coil_wide_coil": "宽型线圈装置",
        "crossbow_top_crossbow": "顶部弩炮",
        "fire_side_crossbow": "侧边弩炮",
        "fire_side_fire": "侧边火炮",
        "fire_top_fire": "顶部火炮",
    }

    for terrain_idx, terrain_key in enumerate(terrain_feature_columns):
        terrain_name = terrain_display_mapping.get(terrain_key, terrain_key)

        for monster_idx in range(1, MONSTER_COUNT + 1):
            monster_name = get_monster_name(monster_idx)

            # 转换数据类型
            df[f"FL{terrain_idx+1}"] = pd.to_numeric(
                df[f"FL{terrain_idx+1}"], errors="coerce"
            ).fillna(0)
            df[f"FR{terrain_idx+1}"] = pd.to_numeric(
                df[f"FR{terrain_idx+1}"], errors="coerce"
            ).fillna(0)
            df[f"L{monster_idx}"] = pd.to_numeric(
                df[f"L{monster_idx}"], errors="coerce"
            ).fillna(0)
            df[f"R{monster_idx}"] = pd.to_numeric(
                df[f"R{monster_idx}"], errors="coerce"
            ).fillna(0)

            # 有地形时的表现
            terrain_left_games = df[
                (df[f"FL{terrain_idx+1}"] == 1) & (df[f"L{monster_idx}"] > 0)
            ]
            terrain_right_games = df[
                (df[f"FR{terrain_idx+1}"] == 1) & (df[f"R{monster_idx}"] > 0)
            ]

            terrain_total = len(terrain_left_games) + len(terrain_right_games)
            if terrain_total < 5:  # 数据量太少
                continue

            terrain_wins = len(
                terrain_left_games[terrain_left_games["Result"] == "L"]
            ) + len(terrain_right_games[terrain_right_games["Result"] == "R"])
            terrain_win_rate = terrain_wins / terrain_total

            # 无地形时的表现
            normal_left_games = df[
                (df[f"FL{terrain_idx+1}"] == 0) & (df[f"L{monster_idx}"] > 0)
            ]
            normal_right_games = df[
                (df[f"FR{terrain_idx+1}"] == 0) & (df[f"R{monster_idx}"] > 0)
            ]

            normal_total = len(normal_left_games) + len(normal_right_games)
            if normal_total < 5:
                continue

            normal_wins = len(
                normal_left_games[normal_left_games["Result"] == "L"]
            ) + len(normal_right_games[normal_right_games["Result"] == "R"])
            normal_win_rate = normal_wins / normal_total

            # 计算影响程度
            effect = terrain_win_rate - normal_win_rate

            if abs(effect) >= 0.05:  # 胜率差异超过5%才记录
                terrain_effects.append(
                    {
                        "地形": terrain_name,
                        "怪物": monster_name,
                        "怪物ID": monster_idx,
                        "地形胜率": terrain_win_rate,
                        "普通胜率": normal_win_rate,
                        "影响程度": effect,
                        "地形场次": terrain_total,
                        "普通场次": normal_total,
                    }
                )

    # 按影响程度绝对值排序
    terrain_effects.sort(key=lambda x: -abs(x["影响程度"]))
    return pd.DataFrame(terrain_effects[:20])  # 增加到前20个


def analyze_device_counter_effects(df):
    """分析五个装置对怪物的克制效果"""
    device_counter_results = {}

    # 获取实际的地形特征列名
    terrain_feature_columns = get_terrain_feature_columns()

    # 定义五个装置类别及其对应的地形特征
    device_categories = {
        "altar": {
            "name": "祭坛",
            "features": [f for f in terrain_feature_columns if "altar" in f],
            "description": "祭坛类装置",
        },
        "block": {
            "name": "箱子/阻挡",
            "features": [f for f in terrain_feature_columns if "block" in f],
            "description": "方块阻挡类装置",
        },
        "coil": {
            "name": "电桩",
            "features": [f for f in terrain_feature_columns if "coil" in f],
            "description": "线圈电桩装置",
        },
        "crossbow": {
            "name": "弩箭",
            "features": [
                f for f in terrain_feature_columns if "crossbow" in f
            ],
            "description": "弩炮装置",
        },
        "fire": {
            "name": "火炮",
            "features": [f for f in terrain_feature_columns if "fire" in f],
            "description": "火炮装置",
        },
    }

    for device_key, device_info in device_categories.items():
        device_name = device_info["name"]
        device_features = device_info["features"]

        if not device_features:
            continue

        device_effects = []

        # 对每个怪物分析该装置的克制效果
        for monster_idx in range(1, MONSTER_COUNT + 1):
            monster_name = get_monster_name(monster_idx)

            # 收集该装置所有特征的统计数据
            total_device_games = 0
            total_device_wins = 0
            total_normal_games = 0
            total_normal_wins = 0

            for terrain_key in device_features:
                if terrain_key not in terrain_feature_columns:
                    continue

                terrain_idx = terrain_feature_columns.index(terrain_key)

                # 转换数据类型
                df[f"FL{terrain_idx+1}"] = pd.to_numeric(
                    df[f"FL{terrain_idx+1}"], errors="coerce"
                ).fillna(0)
                df[f"FR{terrain_idx+1}"] = pd.to_numeric(
                    df[f"FR{terrain_idx+1}"], errors="coerce"
                ).fillna(0)
                df[f"L{monster_idx}"] = pd.to_numeric(
                    df[f"L{monster_idx}"], errors="coerce"
                ).fillna(0)
                df[f"R{monster_idx}"] = pd.to_numeric(
                    df[f"R{monster_idx}"], errors="coerce"
                ).fillna(0)

                # 有该装置时的表现
                device_left_games = df[
                    (df[f"FL{terrain_idx+1}"] == 1)
                    & (df[f"L{monster_idx}"] > 0)
                ]
                device_right_games = df[
                    (df[f"FR{terrain_idx+1}"] == 1)
                    & (df[f"R{monster_idx}"] > 0)
                ]

                device_games_count = len(device_left_games) + len(
                    device_right_games
                )
                device_wins_count = len(
                    device_left_games[device_left_games["Result"] == "L"]
                ) + len(
                    device_right_games[device_right_games["Result"] == "R"]
                )

                # 无该装置时的表现
                normal_left_games = df[
                    (df[f"FL{terrain_idx+1}"] == 0)
                    & (df[f"L{monster_idx}"] > 0)
                ]
                normal_right_games = df[
                    (df[f"FR{terrain_idx+1}"] == 0)
                    & (df[f"R{monster_idx}"] > 0)
                ]

                normal_games_count = len(normal_left_games) + len(
                    normal_right_games
                )
                normal_wins_count = len(
                    normal_left_games[normal_left_games["Result"] == "L"]
                ) + len(
                    normal_right_games[normal_right_games["Result"] == "R"]
                )

                total_device_games += device_games_count
                total_device_wins += device_wins_count
                total_normal_games += normal_games_count
                total_normal_wins += normal_wins_count

            # 计算整体效果
            if (
                total_device_games >= 10 and total_normal_games >= 10
            ):  # 确保有足够的数据
                device_win_rate = total_device_wins / total_device_games
                normal_win_rate = total_normal_wins / total_normal_games
                effect = device_win_rate - normal_win_rate

                # 计算克制程度（负值表示被该装置克制）
                counter_effect = -effect  # 装置对怪物的克制效果

                if abs(effect) >= 0.05:  # 胜率差异超过5%才记录
                    device_effects.append(
                        {
                            "怪物": monster_name,
                            "怪物ID": monster_idx,
                            "装置胜率": device_win_rate,
                            "普通胜率": normal_win_rate,
                            "克制程度": counter_effect,  # 正值表示被该装置克制
                            "装置场次": total_device_games,
                            "普通场次": total_normal_games,
                            "效果类型": (
                                "被克制" if counter_effect > 0 else "克制装置"
                            ),
                        }
                    )

        # 按克制程度排序（被克制程度最高的在前）
        device_effects.sort(key=lambda x: -x["克制程度"])
        device_counter_results[device_key] = {
            "name": device_name,
            "description": device_info["description"],
            "features": device_features,
            "effects": device_effects[:10],  # 取前10个被克制最严重的怪物
        }

    return device_counter_results


def create_html_table(
    df, columns, title, is_combo=False, monster_relations=None
):
    """创建带有怪物头像的HTML表格"""
    html = f"<h2>{title}</h2>\n<table>\n<tr>"

    # 表头
    if is_combo:
        html += "<th>组合</th>"
    else:
        html += "<th>怪物</th>"

    for col in columns:
        html += f"<th>{col}</th>"

    # 如果是胜率表且有关系数据，添加额外的列
    if not is_combo and monster_relations and title == "所有怪物胜率排行榜":
        html += "<th>最佳队友</th><th>克制</th><th>被克制</th>"

    html += "</tr>\n"

    # 表格内容
    row_number = 1
    for idx, row in df.iterrows():
        html += "<tr>"

        # 怪物图片和名称
        monster_id = None
        if is_combo and "ID1" in row and "ID2" in row:
            monster1_name = get_monster_name(row["ID1"])
            monster2_name = get_monster_name(row["ID2"])
            monster1_orig = get_monster_original_name(row["ID1"])
            monster2_orig = get_monster_original_name(row["ID2"])
            html += f"""<td>
                <span style="font-size:16px;font-weight:bold;color:#4CAF50;margin-right:8px;">{row_number}.</span>
                {monster_img_tag(monster1_orig, 30)}
                <span>{monster1_name}</span><br>
                {monster_img_tag(monster2_orig, 30)}
                <span>{monster2_name}</span>
            </td>"""
        elif "怪物ID" in row:
            monster_name = get_monster_name(row["怪物ID"])
            monster_orig = get_monster_original_name(row["怪物ID"])
            display_name = row.get("怪物", monster_name)
            monster_id = row["怪物ID"]
            html += f"""<td>
                <span style="font-size:16px;font-weight:bold;color:#4CAF50;margin-right:8px;">{row_number}.</span>
                {monster_img_tag(monster_orig, 50)}
                <span style="font-weight:bold;">{display_name}</span>
            </td>"""
        else:
            # 对于胜率表，使用索引作为怪物名称
            monster_name = idx
            # 尝试从怪物数据中获取ID
            monster_orig = monster_name
            for mid in range(1, MONSTER_COUNT + 1):
                if get_monster_name(mid) == monster_name:
                    monster_id = mid
                    monster_orig = get_monster_original_name(mid)
                    break
            html += f"""<td>
                <span style="font-size:16px;font-weight:bold;color:#4CAF50;margin-right:8px;">{row_number}.</span>
                {monster_img_tag(monster_orig, 50)}
                <span style="font-weight:bold;">{monster_name}</span>
            </td>"""

        # 数据列
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                if "率" in col or "程度" in col:
                    html += f"<td>{value:.2%}</td>"
                else:
                    html += f"<td>{value:.2f}</td>"
            else:
                html += f"<td>{value}</td>"

        # 如果是胜率表且有关系数据，添加关系信息
        if (
            not is_combo
            and monster_relations
            and title == "所有怪物胜率排行榜"
            and monster_id
        ):
            relations = monster_relations.get(int(monster_id), {})

            # 最佳队友
            html += "<td>"
            if (
                relations
                and "best_teammates" in relations
                and relations["best_teammates"]
            ):
                teammates = []
                for teammate in relations["best_teammates"]:
                    teammate_orig = get_monster_original_name(
                        teammate["partner_id"]
                    )
                    teammates.append(f"""<div style="margin:3px 0;">
                        {monster_img_tag(teammate_orig, 20)}
                        <small>{teammate['partner_name']} ({teammate['lift']:.2f}x)</small>
                    </div>""")
                html += "".join(teammates)
            else:
                html += "<small style='color:#888;'>暂无数据</small>"
            html += "</td>"

            # 克制关系
            html += "<td>"
            if relations and "counters" in relations and relations["counters"]:
                counters = []
                for counter in relations["counters"]:
                    counter_orig = get_monster_original_name(
                        counter["opponent_id"]
                    )
                    counters.append(f"""<div style="margin:3px 0;">
                        {monster_img_tag(counter_orig, 20)}
                        <small>{counter['opponent_name']} ({counter['win_rate']:.0%})</small>
                    </div>""")
                html += "".join(counters)
            else:
                html += "<small style='color:#888;'>暂无数据</small>"
            html += "</td>"

            # 被克制关系
            html += "<td>"
            if (
                relations
                and "countered_by" in relations
                and relations["countered_by"]
            ):
                countered = []
                for counter in relations["countered_by"]:
                    counter_orig = get_monster_original_name(
                        counter["opponent_id"]
                    )
                    countered.append(f"""<div style="margin:3px 0;">
                        {monster_img_tag(counter_orig, 20)}
                        <small>{counter['opponent_name']} ({counter['lose_rate']:.0%})</small>
                    </div>""")
                html += "".join(countered)
            else:
                html += "<small style='color:#888;'>暂无数据</small>"
            html += "</td>"

        html += "</tr>\n"
        row_number += 1

    html += "</table>\n"
    return html


def create_device_counter_html(device_counter_effects):
    """创建装置克制效果的HTML表格"""
    html = "<h2>装置克制效果统计</h2>\n"

    for device_key, device_data in device_counter_effects.items():
        if not device_data["effects"]:
            continue

        device_name = device_data["name"]
        device_description = device_data["description"]
        effects = device_data["effects"]

        html += f"<h3>{device_name}（{device_description}）</h3>\n"
        html += "<table>\n<tr>"
        html += "<th>怪物</th><th>克制程度</th><th>装置胜率</th><th>普通胜率</th><th>装置场次</th><th>普通场次</th><th>效果类型</th>"
        html += "</tr>\n"

        row_number = 1
        for effect in effects:
            monster_name = effect["怪物"]
            monster_orig = get_monster_original_name(effect["怪物ID"])
            html += f"""<tr>
                <td>
                    <span style="font-size:16px;font-weight:bold;color:#4CAF50;margin-right:8px;">{row_number}.</span>
                    {monster_img_tag(monster_orig, 40)}
                    <span style="font-weight:bold;">{monster_name}</span>
                </td>
                <td style="color: {'red' if effect['克制程度'] > 0 else 'green'};">{effect['克制程度']:.2%}</td>
                <td>{effect['装置胜率']:.2%}</td>
                <td>{effect['普通胜率']:.2%}</td>
                <td>{effect['装置场次']}</td>
                <td>{effect['普通场次']}</td>
                <td style="color: {'red' if effect['效果类型'] == '被克制' else 'green'};">{effect['效果类型']}</td>
            </tr>\n"""
            row_number += 1

        html += "</table>\n<br>\n"

    return html


def generate_comprehensive_report():
    """生成综合统计报告"""
    print("正在加载数据...")
    df = load_data()

    print("正在计算怪物胜率...")
    win_rates = calculate_all_monster_win_rates(df)

    print("正在分析怪物配合...")
    combinations = analyze_monster_combinations(df)

    print("正在分析被克制关系...")
    countered = find_countered_monsters(df)

    if FIELD_FEATURE_COUNT > 0:
        print("正在分析地形效果...")
        terrain_effects = analyze_terrain_effects(df)

        print("正在分析装置克制效果...")
        device_counter_effects = analyze_device_counter_effects(df)
    else:
        print("地形特征数量为0，跳过地形分析...")
        terrain_effects = pd.DataFrame()
        device_counter_effects = {}

    print("正在分析个体怪物关系...")
    monster_relations = analyze_individual_monster_relations(df)

    # 创建HTML报告
    total_battles = len(df)
    monster_count = MONSTER_COUNT
    field_count = FIELD_FEATURE_COUNT

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #4CAF50;
            border-left: 4px solid #4CAF50;
            padding-left: 10px;
            margin-top: 30px;
        }}
        table {{ 
            border-collapse: collapse; 
            margin: 20px 0; 
            width: 100%;
            background-color: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        th, td {{ 
            border: 1px solid #ddd; 
            padding: 12px; 
            text-align: left; 
        }}
        th {{ 
            background-color: #4CAF50; 
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        img {{ 
            width: 30px; 
            height: 30px; 
            vertical-align: middle; 
            margin-right: 5px;
            border-radius: 4px;
        }}
        .stats {{
            background-color: #e8f5e8;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>明日方舟争锋频道绿藤城</h1>
    <div class="stats">
        <p><strong>数据概览：</strong></p>
        <ul>
            <li>总战斗记录：{total_battles} 场</li>
            <li>统计怪物数量：{monster_count} 种</li>
            <li>地形特征数量：{field_count} 种</li>
        </ul>
    </div>
"""

    # 1. 所有怪物胜率
    if not win_rates.empty:
        html += create_html_table(
            win_rates,
            ["胜场", "总场数", "胜率", "参战率"],
            "所有怪物胜率排行榜",
            monster_relations=monster_relations,
        )
    else:
        html += "<h2>所有怪物胜率排行榜</h2><p>暂无数据</p>"

    # 2. 最佳配合
    if not combinations.empty:
        html += create_html_table(
            combinations.head(20),
            ["提升度", "组合胜率", "出场次数"],
            "最佳怪物配合TOP20",
            is_combo=True,
        )
    else:
        html += "<h2>最佳怪物配合</h2><p>暂无足够的配合数据</p>"

    # 4. 地形效果
    if not terrain_effects.empty:
        html += create_html_table(
            terrain_effects,
            ["地形", "地形胜率", "普通胜率", "影响程度"],
            "地形影响最大的怪物TOP20",
        )
    else:
        html += "<h2>地形影响</h2><p>暂无足够的地形数据</p>"

    # 5. 装置克制效果
    if device_counter_effects:
        html += create_device_counter_html(device_counter_effects)
    else:
        html += "<h2>装置克制效果统计</h2><p>暂无足够的装置数据</p>"

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    html += f"""
    <div class="stats">
        <p><em>报告生成时间：{timestamp}</em></p>
        <p><em>注：数据基于历史战斗记录，仅供参考</em></p>
    </div>
</body>
</html>
"""

    # 保存报告
    with open("comprehensive_monster_report.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("统计报告已生成：comprehensive_monster_report.html")

    # 也输出到控制台
    print("\n=== 怪物胜率TOP10 ===")
    print(win_rates.head(10).to_string())

    if not combinations.empty:
        print("\n=== 最佳配合TOP10 ===")
        print(
            combinations.head(10)[
                ["组合", "提升度", "组合胜率", "出场次数"]
            ].to_string()
        )

    # 输出装置克制效果
    if device_counter_effects:
        print("\n=== 装置克制效果统计 ===")
        for device_key, device_data in device_counter_effects.items():
            if device_data["effects"]:
                print(
                    f"\n{device_data['name']}（{device_data['description']}）最克制的怪物TOP5："
                )
                for i, effect in enumerate(device_data["effects"][:5]):
                    print(
                        f"  {i+1}. {effect['怪物']} - 克制程度: {effect['克制程度']:.2%} ({effect['效果类型']})"
                    )

    if not terrain_effects.empty:
        print("\n=== 地形影响TOP10 ===")
        print(
            terrain_effects.head(10)[
                ["地形", "怪物", "影响程度", "地形胜率", "普通胜率"]
            ].to_string()
        )


if __name__ == "__main__":
    generate_comprehensive_report()
