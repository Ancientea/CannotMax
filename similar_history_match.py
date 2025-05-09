import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class HistoryMatch:
    """错题本数据集的读取和处理类"""

    def __init__(self):
        self.past_left = None
        self.past_right = None
        self.labels = None
        self.feat_past = None
        self.N_history = None

        # 初始化历史记录缓存
        self.last_try = False
        self.last_left = None
        self.last_right = None
        self.swap = None
        self.top20_idx = None

        # 读取数据集
        self.load_history_data()

    def __len__(self):
        return self.N_history

    def load_history_data(self):
        """错题本读取的数据集，只在 __init__ 里启动时调用"""
        try:
            df = pd.read_csv(r"arknights.csv", header=None, skiprows=1)
            self.past_left = df.iloc[:, 0:56].to_numpy(float)
            self.past_right = df.iloc[:, 56:112].to_numpy(float)
            self.labels = df.iloc[:, 112].to_numpy()
        except Exception as e:
            self.past_left = np.zeros((0, 56), dtype=float)
            self.past_right = np.zeros((0, 56), dtype=float)
            self.labels = np.zeros((0,), dtype=float)
        # 组合特征
        self.feat_past = np.hstack(
            [self.past_left + self.past_right, np.abs(self.past_left - self.past_right)]
        )
        self.N_history = len(self.past_left)

    def render_similar_matches(self, left_counts: np.typing.ArrayLike, right_counts: np.typing.ArrayLike):
        if not (self.last_try and np.all(left_counts == self.last_left) and np.all(right_counts == self.last_right)):
            self.last_left, self.last_right = left_counts, right_counts
            self.last_try = True
            self.search_similar_matches(left_counts, right_counts)
            print("错题本匹配成功")
        self.render_matches(left_counts, right_counts)

    def search_similar_matches(self, left_counts: np.typing.ArrayLike, right_counts: np.typing.ArrayLike):
        try:
            cur_left = left_counts
            cur_right = right_counts

            setL_cur = set(np.where(cur_left > 0)[0])
            setR_cur = set(np.where(cur_right > 0)[0])

            # 相似度和特征
            feat_cur = np.hstack([cur_left + cur_right, np.abs(cur_left - cur_right)])
            feat_cur = feat_cur.reshape(1, -1)
            sims = cosine_similarity(feat_cur, self.feat_past)[0]  # shape (N_history,)

            N = self.N_history
            # 数组
            cats = np.empty(N, np.int8)
            qdiff_other = np.empty(N, np.int16)
            match_other = np.empty(N, np.int16)
            swap = np.zeros(N, dtype=bool)

            # 分类函数
            def classify(typeL_eq, typeR_eq, cntL_eq, cntR_eq):
                if typeL_eq and typeR_eq and cntL_eq and cntR_eq:
                    return 0
                if typeL_eq and typeR_eq:
                    return 1 if (cntL_eq or cntR_eq) else 2
                if (typeL_eq and cntL_eq) or (typeR_eq and cntR_eq):
                    return 3
                if typeL_eq or typeR_eq:
                    return 4
                return 5

            # 逻辑
            for i in range(N):
                Lraw, Rraw = self.past_left[i], self.past_right[i]

                # 判断要不要镜像
                missA = len(setL_cur ^ set(np.where(Lraw > 0)[0])) + len(
                    setR_cur ^ set(np.where(Rraw > 0)[0])
                )
                cntA = int(np.abs(Lraw - cur_left).sum() + np.abs(Rraw - cur_right).sum())

                missB = len(setL_cur ^ set(np.where(Rraw > 0)[0])) + len(
                    setR_cur ^ set(np.where(Lraw > 0)[0])
                )
                cntB = int(np.abs(Rraw - cur_left).sum() + np.abs(Lraw - cur_right).sum())

                if (missB, cntB) < (missA, cntA):
                    swap[i] = True
                    Lh, Rh = Rraw, Lraw
                else:
                    Lh, Rh = Lraw, Rraw

                need_L = np.where(cur_left > 0)[0]
                need_R = np.where(cur_right > 0)[0]
                setL_h = set(np.where(Lh > 0)[0])
                setR_h = set(np.where(Rh > 0)[0])

                full_L = np.all(Lh[need_L] == cur_left[need_L])
                full_R = np.all(Rh[need_R] == cur_right[need_R])

                diff_L = int(np.abs(Lh[need_L] - cur_left[need_L]).sum())
                diff_R = int(np.abs(Rh[need_R] - cur_right[need_R]).sum())

                hit_other_R = len(setR_h & set(need_R))
                hit_other_L = len(setL_h & set(need_L))
                match_other[i] = min(hit_other_L, hit_other_R)
                if hit_other_R and not full_R:
                    qdiff_other[i] = diff_R
                elif hit_other_L and not full_L:
                    qdiff_other[i] = diff_L
                else:
                    qdiff_other[i] = 0

                # 分类
                cats[i] = classify(
                    setL_h == setL_cur,
                    setR_h == setR_cur,
                    np.array_equal(Lh, cur_left),
                    np.array_equal(Rh, cur_right),
                )

            # 排序
            order = np.lexsort((-sims, qdiff_other, -match_other, cats))
            good = order[match_other[order] > 0]
            backup = order[match_other[order] == 0]

            # 确保按照相似度降序排列
            top20_idx = np.concatenate((good, backup))[:20]
            # 按相似度重新排序
            self.top20_idx = top20_idx[np.argsort(-sims[top20_idx])]
            self.swap = swap
            self.cur_left = cur_left
            self.cur_right = cur_right
            self.sims = sims
            self.swap = swap
        except Exception as e:
            print("[匹配错题本失败]", e)

    def render_matches(self, left_counts: np.typing.ArrayLike, right_counts: np.typing.ArrayLike):
        try:
            cur_left = left_counts
            cur_right = right_counts
            top5_idx = self.top20_idx[:5]

            # 胜率计算和标题渲染
            tgtL = max(
                (i for i, v in enumerate(cur_left) if v > 0), key=cur_left.__getitem__, default=None
            )
            tgtR = max(
                (i for i, v in enumerate(cur_right) if v > 0),
                key=cur_right.__getitem__,
                default=None,
            )

            lw = rw = 0
            for idx in top5_idx:
                lab = self.labels[idx]
                Lh, Rh = self.past_left[idx], self.past_right[idx]
                if self.swap[idx]:
                    lab = "L" if lab == "R" else "R"
                    Lh, Rh = Rh, Lh
                if tgtL is not None:
                    side = "L" if Lh[tgtL] > 0 else "R"
                    lw += lab == side
                if tgtR is not None:
                    side = "L" if Lh[tgtR] > 0 else "R"
                    rw += lab == side
            left_rate = lw / len(top5_idx) if top5_idx.size else 0
            right_rate = rw / len(top5_idx) if top5_idx.size else 0

            self.left_rate = left_rate
            self.right_rate = right_rate
        except Exception as e:
            print("[渲染错题本失败]", e)
