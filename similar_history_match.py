import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from config import MONSTER_COUNT
from config import FIELD_FEATURE_COUNT

class HistoryMatch:
    """错题本数据集的读取和处理类"""

    def __init__(self, csv_path="arknights.csv"):
        # 初始化时加载历史对局数据
        self.csv_path = csv_path
        self.load_history_data()

    def __len__(self):
        # 返回历史对局数量
        return self.N_history

    def load_history_data(self):
        """读取 CSV 文件，加载历史对局的左右阵容、地形及胜负标签"""
        try:
            df = pd.read_csv(self.csv_path, header=None, skiprows=1)
            
            # 新数据格式: [怪物L(77), 场地L(6), 怪物R(77), 场地R(6), Result, ImgPath]
            total_features = (MONSTER_COUNT + FIELD_FEATURE_COUNT) * 2
            
            if df.shape[1] >= total_features + 1:  # 至少包含特征和结果列
                # 提取各部分特征
                left_monster_end = MONSTER_COUNT
                left_field_end = MONSTER_COUNT + FIELD_FEATURE_COUNT
                right_monster_end = MONSTER_COUNT + FIELD_FEATURE_COUNT + MONSTER_COUNT
                right_field_end = MONSTER_COUNT + FIELD_FEATURE_COUNT + MONSTER_COUNT + FIELD_FEATURE_COUNT
                
                # 分别提取怪物和地形特征
                left_monsters = df.iloc[:, 0:left_monster_end].values.astype(float)
                left_terrain = df.iloc[:, left_monster_end:left_field_end].values.astype(float)
                right_monsters = df.iloc[:, left_field_end:right_monster_end].values.astype(float)
                right_terrain = df.iloc[:, right_monster_end:right_field_end].values.astype(float)
                
                # 合并怪物特征（只使用怪物部分进行相似度计算）
                self.past_left = left_monsters
                self.past_right = right_monsters
                
                # 保存地形特征用于显示
                self.past_left_terrain = left_terrain
                self.past_right_terrain = right_terrain
                
                # 胜负标签
                self.labels = df.iloc[:, total_features].values
            else:
                # 兼容旧格式：只有怪物特征
                self.past_left = df.iloc[:, 0:MONSTER_COUNT].values.astype(float)
                self.past_right = df.iloc[:, MONSTER_COUNT:MONSTER_COUNT*2].values.astype(float)
                self.labels = df.iloc[:, MONSTER_COUNT*2].values
                
                # 地形特征为空
                self.past_left_terrain = np.zeros((len(self.past_left), FIELD_FEATURE_COUNT))
                self.past_right_terrain = np.zeros((len(self.past_right), FIELD_FEATURE_COUNT))
                
        except Exception as e:
            print(f"加载历史数据失败: {e}")
            # 加载失败时，初始化为空数组
            self.past_left = np.zeros((0, MONSTER_COUNT), float)
            self.past_right = np.zeros((0, MONSTER_COUNT), float)
            self.past_left_terrain = np.zeros((0, FIELD_FEATURE_COUNT), float)
            self.past_right_terrain = np.zeros((0, FIELD_FEATURE_COUNT), float)
            self.labels = np.array([], dtype=str)

        # 构造历史对局特征: 左右数量之和与差的绝对值拼接（只使用怪物特征）
        self.feat_past = np.hstack([
            self.past_left + self.past_right,
            np.abs(self.past_left - self.past_right)
        ])
        # 历史对局总数
        self.N_history = self.past_left.shape[0]

    def render_similar_matches(self, left_counts: np.ndarray, right_counts: np.ndarray):
        """返回与当前对局最相似的历史对局索引及胜率统计"""
        # 将输入转为浮点型数组
        cur_left = left_counts.astype(float)
        cur_right = right_counts.astype(float)

        # 计算当前存在的兵种布尔向量
        pres_L = cur_left > 0
        pres_R = cur_right > 0
        need_L_idx = np.nonzero(pres_L)[0]  # 当前左侧有兵的索引
        need_R_idx = np.nonzero(pres_R)[0]  # 当前右侧有兵的索引

        # 构造当前对局特征并计算与所有历史的余弦相似度
        feat_cur = np.hstack([cur_left + cur_right, np.abs(cur_left - cur_right)]).reshape(1, -1)
        sims = cosine_similarity(feat_cur, self.feat_past)[0]

        # 历史对局的存在布尔矩阵
        hist_pres_L = self.past_left > 0  # shape (N_history, MONSTER_COUNT)
        hist_pres_R = self.past_right > 0

        # 计算未镜像(missA, cntA)和镜像后(missB, cntB)的缺兵及数量差距
        missA = np.sum(np.logical_xor(pres_L, hist_pres_L), axis=1) + np.sum(
            np.logical_xor(pres_R, hist_pres_R), axis=1)
        cntA = np.sum(np.abs(self.past_left - cur_left), axis=1) + np.sum(
            np.abs(self.past_right - cur_right), axis=1)

        missB = np.sum(np.logical_xor(pres_L, hist_pres_R), axis=1) + np.sum(
            np.logical_xor(pres_R, hist_pres_L), axis=1)
        cntB = np.sum(np.abs(self.past_right - cur_left), axis=1) + np.sum(
            np.abs(self.past_left - cur_right), axis=1)

        # 根据(miss, cnt)比较，决定是否对历史数据做镜像处理
        swap = (missB < missA) | ((missB == missA) & (cntB < cntA))

        # 生成镜像后的历史左右阵容
        Lh = np.where(swap[:, None], self.past_right, self.past_left)
        Rh = np.where(swap[:, None], self.past_left, self.past_right)

        # 修复bug：3B对阵3A在csv文件中，但输入3A、3B不返回，输入3B、3A才返回
        # 使用镜像后的阵容构造存在布尔矩阵（后续匹配统计应基于镜像后的阵容）
        hist_pres_Lh = Lh > 0  # shape (N_history, MONSTER_COUNT)
        hist_pres_Rh = Rh > 0

        # 判断在需求索引上是否完全匹配
        full_L = np.all(Lh[:, need_L_idx] == cur_left[need_L_idx], axis=1)
        full_R = np.all(Rh[:, need_R_idx] == cur_right[need_R_idx], axis=1)

        # 计算需求索引处的数量差和
        diff_L = np.sum(np.abs(Lh[:, need_L_idx] - cur_left[need_L_idx]), axis=1)
        diff_R = np.sum(np.abs(Rh[:, need_R_idx] - cur_right[need_R_idx]), axis=1)

        # 计算对手兵种在本方需求中的命中数，取最小值作为 match_other
        hit_L = np.sum(hist_pres_Rh[:, need_L_idx] & pres_L[need_L_idx], axis=1)
        hit_R = np.sum(hist_pres_Lh[:, need_R_idx] & pres_R[need_R_idx], axis=1)
        match_other = np.minimum(hit_L, hit_R)

        # 根据命中侧及是否完全匹配，选择对应的 qdiff_other
        qdiff_other = np.where(
            (hit_R > 0) & (~full_R), diff_R,
            np.where((hit_L > 0) & (~full_L), diff_L, 0)
        )

        # 批量计算分类所需的布尔向量
        # 注意：类型（presence）比较也应基于镜像后的阵容
        typeL_eq = np.all(hist_pres_Lh == pres_L, axis=1)
        typeR_eq = np.all(hist_pres_Rh == pres_R, axis=1)
        cntL_eq = np.all(Lh == cur_left, axis=1)
        cntR_eq = np.all(Rh == cur_right, axis=1)

        # 初始化类别为最松散的 5（默认）
        cats = np.full(self.N_history, 5, dtype=np.int8)

        # 重新定义分类优先级（数值越小优先级越高）：
        # 0: 双方种类与数量都完全相同
        # 1: 双方种类相同，数量均不同但成比例（如 1A1B vs 2A2B）
        # 2: 双方种类相同，且至少一侧数量相同（但不是双方都相同）
        # 3: 双方种类相同，但双方数量均不同（且不成比例）
        # 5: 其它（默认）
        same_species = typeL_eq & typeR_eq

        mask0 = same_species & cntL_eq & cntR_eq
        mask1 = same_species & (cntL_eq | cntR_eq) & ~mask0

        # 检测“成比例”:
        # 在双方各自非零位置上，Lh/cur_left 与 Rh/cur_right 分别行内常数，
        # 且左右两侧比例相同，且比例不为 1（确保“数量均不同”）
        # 注意：当某侧不存在任意单位时，认为该侧比例为 1 且恒定
        if need_L_idx.size > 0:
            ratios_L = Lh[:, need_L_idx] / np.maximum(cur_left[need_L_idx], 1e-12)
            rL_min = ratios_L.min(axis=1)
            rL_max = ratios_L.max(axis=1)
            uniform_L = np.isclose(rL_min, rL_max, rtol=1e-3, atol=1e-6)
            rL = 0.5 * (rL_min + rL_max)
        else:
            uniform_L = np.ones(self.N_history, dtype=bool)
            rL = np.ones(self.N_history, dtype=float)

        if need_R_idx.size > 0:
            ratios_R = Rh[:, need_R_idx] / np.maximum(cur_right[need_R_idx], 1e-12)
            rR_min = ratios_R.min(axis=1)
            rR_max = ratios_R.max(axis=1)
            uniform_R = np.isclose(rR_min, rR_max, rtol=1e-3, atol=1e-6)
            rR = 0.5 * (rR_min + rR_max)
        else:
            uniform_R = np.ones(self.N_history, dtype=bool)
            rR = np.ones(self.N_history, dtype=float)

        same_ratio = np.isclose(rL, rR, rtol=1e-3, atol=1e-6)
        ratio_not_one = ~np.isclose(rL, 1.0, rtol=1e-3, atol=1e-6)  # rL==rR 时即可代表两侧都不为1
        proportional = uniform_L & uniform_R & same_ratio & ratio_not_one

        # 2类：同种类，数量均不同且成比例
        mask2 = same_species & (~cntL_eq) & (~cntR_eq) & proportional
        # 3类：同种类，数量均不同但不成比例
        mask3 = same_species & (~cntL_eq) & (~cntR_eq) & (~proportional)

        cats[mask0] = 0
        cats[mask1] = 2
        cats[mask2] = 1
        cats[mask3] = 3

        # 使用 lexsort 按 (-sims, qdiff_other, -match_other, cats) 排序
        order = np.lexsort((-sims, qdiff_other, -match_other, cats))
        good = order[match_other[order] > 0]
        backup = order[match_other[order] == 0]
        top20 = np.concatenate([good, backup])[:20]
        # 移除最终“仅按相似度”的重排，保持 cats 优先级
        # top20 = top20[np.argsort(-sims[top20])]
        self.top20_idx = top20

        # 从前5条中计算左右胜率
        top5 = top20[:5]
        labs = np.where(swap[top5], np.where(self.labels[top5]=="L", "R", "L"), self.labels[top5])
        tgtL = need_L_idx[np.argmax(cur_left[need_L_idx])] if need_L_idx.size else None
        tgtR = need_R_idx[np.argmax(cur_right[need_R_idx])] if need_R_idx.size else None

        lw = np.sum([lab == ("L" if (Lh[i, tgtL] if tgtL is not None else 0) > 0 else "R")
                     for i, lab in zip(top5, labs)])
        rw = np.sum([lab == ("L" if (Lh[i, tgtR] if tgtR is not None else 0) > 0 else "R")
                     for i, lab in zip(top5, labs)])
        self.left_rate = lw / len(top5) if top5.size else 0
        self.right_rate = rw / len(top5) if top5.size else 0
        self.sims = sims
        self.swap = swap
        self.cur_left = cur_left
        self.cur_right = cur_right
        return self.top20_idx, self.left_rate, self.right_rate

    def get_terrain_names(self, idx, is_swapped=False):
        """获取指定历史对局的地形名称"""
        if idx >= len(self.past_left_terrain):
            return "无地形"
        
        # 根据是否镜像选择地形特征
        terrain_features = self.past_right_terrain[idx] if is_swapped else self.past_left_terrain[idx]
        
        # 获取激活的地形特征索引
        active_indices = np.where(terrain_features > 0)[0]
        
        if len(active_indices) == 0:
            return "无地形"
        
        # 尝试从FieldRecognizer获取实际的特征列名称
        try:
            from field_recognition import FieldRecognizer
            field_recognizer = FieldRecognizer()
            if field_recognizer.is_ready():
                feature_columns = field_recognizer.get_feature_columns()
                # 根据实际特征列名称生成简洁名称
                active_terrains = []
                for i in active_indices:
                    if i < len(feature_columns):
                        full_name = feature_columns[i]
                        # 简化名称映射
                        if "middle_row_blocks" in full_name:
                            simple_name = "中路阻挡"
                        elif "side_fire_cannon_crossbow" in full_name:
                            simple_name = "侧边弩箭"
                        elif "side_fire_cannon_fire" in full_name:
                            simple_name = "侧边火炮"
                        elif "top_crossbow" in full_name:
                            simple_name = "顶部弩箭"
                        elif "top_fire_cannon" in full_name:
                            simple_name = "顶部火炮"
                        elif "two_row_blocks" in full_name:
                            simple_name = "双行阻挡"
                        else:
                            # 如果无法识别，使用原名称的简化版本
                            simple_name = full_name.replace("_", "")
                        active_terrains.append(simple_name)
                
                return "+".join(active_terrains) if active_terrains else "无地形"
        except Exception:
            pass
        
        # 备用硬编码映射（如果无法获取FieldRecognizer）
        terrain_names = {
            0: "中路阻挡",
            1: "侧边弩箭", 
            2: "侧边火炮",
            3: "顶部弩箭",
            4: "顶部火炮",
            5: "双行阻挡"
        }
        
        # 获取所有激活地形的名称
        active_terrains = []
        for i in active_indices:
            if i < len(terrain_names):
                active_terrains.append(terrain_names[i])
        
        # 如果有多个地形，用"+"连接
        return "+".join(active_terrains) if active_terrains else "无地形"
