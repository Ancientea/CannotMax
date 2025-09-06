from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time
import os
import copy
from enum import Enum, auto
import tkinter as tk
# from tkinter import messagebox # messagebox 已被自定义提示替代，可以注释或移除
from PIL import Image, ImageTk
import math
from simulator.battle_field import Battlefield  # 确保 Battlefield 已导入
from simulator.monsters import MonsterFactory  # 确保 MonsterFactory 已导入
from simulator.utils import MONSTER_MAPPING, REVERSE_MONSTER_MAPPING, Faction  # 根据你的实际路径调整
from simulator.vector2d import FastVector  # 确保 FastVector 已导入
from unit import Unit  # 确保 Unit 已导入
import json  # REMOVED_TEAM_INTERFACE: Added missing import for the main block
import random  # REMOVED_TEAM_INTERFACE: Added missing import for the main block
import sys  # Import sys for stdin
from simulator.monsters import AttackState, Monster, MonsterFactory
from recognize import MONSTER_COUNT


class AppState(Enum):
    INITIAL = auto()  # 初始状态
    SETUP = auto()  # 部署阶段
    SIMULATING = auto()  # 模拟运行中
    PAUSED = auto()  # 暂停状态
    ENDED = auto()  # 战斗结束


class StateMachine:
    def __init__(self, ui_update_callback):
        self.state = AppState.INITIAL
        self.ui_update = ui_update_callback

    def transition_to(self, new_state):
        """状态转换并触发UI更新"""
        allowed_transitions = {
            AppState.INITIAL: [AppState.INITIAL, AppState.SETUP, AppState.ENDED],
            AppState.SETUP: [AppState.INITIAL, AppState.SETUP, AppState.SIMULATING],
            AppState.SIMULATING: [AppState.PAUSED, AppState.ENDED],
            AppState.PAUSED: [AppState.SIMULATING, AppState.SETUP],
            AppState.ENDED: [AppState.INITIAL, AppState.SETUP]
        }

        if new_state in allowed_transitions[self.state]:
            self.state = new_state
            self.ui_update()
        else:
            print(f"Illegal state transition from {self.state} to {new_state}")

    def get_control_states(self):
        """返回各控件的状态字典"""
        states = {
            'deploy': {'state': tk.NORMAL, 'text': '部署怪物'},
            'confirm_start': {'state': tk.DISABLED},
            'pause': {'state': tk.DISABLED, 'text': '暂停'},
            'restore': {'state': tk.DISABLED},
            'speed_entry': {'state': tk.NORMAL},
            'clear': {'state': tk.NORMAL},
            'timer': {'text': ''}
        }

        if self.state == AppState.INITIAL:
            pass

        elif self.state == AppState.SETUP:
            states.update({
                'deploy': {'state': tk.NORMAL, 'text': '重新部署'},
                'confirm_start': {'state': tk.NORMAL},
                'timer': {'text': '未开始'}
            })

        elif self.state == AppState.SIMULATING:
            states.update({
                'deploy': {'state': tk.DISABLED, 'text': '重新部署'},
                'confirm_start': {'state': tk.DISABLED},
                'pause': {'state': tk.NORMAL, 'text': '暂停'},
                'clear': {'state': tk.DISABLED},
            })

        elif self.state == AppState.PAUSED:
            states.update({
                'restore': {'state': tk.NORMAL},
                'pause': {'state': tk.NORMAL, 'text': '继续'},
                'deploy': {'state': tk.NORMAL, 'text': '重新部署'}
            })

        elif self.state == AppState.ENDED:
            states.update({
                'deploy': {'state': tk.DISABLED, 'text': '重新部署'},
                'restore': {'state': tk.NORMAL},
                'pause': {'state': tk.DISABLED, 'text': '暂停'},
                'timer': {'text': '战斗结束'}
            })
        return states


class SandboxSimulator:
    def __init__(self, master: tk.Tk, battle_data):
        self.master = master
        self.master.title("沙盒模拟器")
        self.num_monsters = MONSTER_COUNT  # 根据你的怪物总数调整

        self.battle_data = battle_data  # 初始化时传入的怪物配置

        self.grid_width = 13
        self.grid_height = 9
        self.cell_size = 80
        self.canvas_width = self.grid_width * self.cell_size
        self.canvas_height = self.grid_height * self.cell_size

        self.units = []  # UI 对应的单位列表
        self.selected_team = None
        self.selected_unit_id = None

        self.simulation_id = None
        self.speed_multiplier = 10  # 默认速度
        self.is_paused = False  # 新增：暂停状态

        self.monster_data = []  # 在 load_assets 中加载
        self.load_assets()  # 先加载资源，特别是 self.monster_data

        self.battle_field = Battlefield(self.monster_data)  # 核心战场逻辑对象

        self.message_label = None  # 用于显示提示信息的标签
        self.message_timer_id = None  # 用于定时清除提示信息的ID

        self.monte_carlo_mode = True  # 是否启用蒙特卡洛模式
        self.executor = None
        self.mc_running = False
        self.mc_stop_event = threading.Event()
        self.mc_lock = threading.Lock()
        self._mc_counts = Counter()
        self._mc_total_done = 0
        self._mc_total_target = 0
        self._mc_futures = []
        self._mc_result_queue = queue.Queue()  # ← future 完成后把结果丢进来

        self.create_widgets()
        self.master.protocol("WM_DELETE_WINDOW", self.hide_window)
        self.state_machine = StateMachine(self.update_ui_state)
        self.state_machine.transition_to(AppState.INITIAL)
        self.enter_setup_phase()

    def update_ui_state(self):
        """根据当前状态更新所有控件状态"""
        states = self.state_machine.get_control_states()

    def hide_window(self):
        if self.state_machine.state == AppState.SIMULATING:
            self.master.after_cancel(self.simulation_id)
            self.state_machine.transition_to(AppState.PAUSED)
        self.master.destroy()

    def load_assets(self):
        self.icons = {}
        try:
            with open("simulator/monsters.json", encoding='utf-8') as f:
                self.monster_data = json.load(f)["monsters"]
        except FileNotFoundError:
            print("错误: monsters.json 未找到，请检查路径！")
            self.monster_data = []

            return

        for i in range(self.num_monsters):
            image_file_id = i + 1
            try:
                image = Image.open(f'images/{image_file_id}.png')
                self.icons[i] = {
                    "red": ImageTk.PhotoImage(image.resize((40, 40))),
                    "blue": ImageTk.PhotoImage(image.resize((40, 40)).transpose(Image.FLIP_LEFT_RIGHT))
                }
            except Exception as e:
                # 同样，show_message_below_button 可能还不可用
                print(f"加载图标错误 (图标键: {i}, 文件名ID: {image_file_id}): {str(e)}")
                self.icons[i] = {
                    "red": ImageTk.PhotoImage(Image.new("RGB", (40, 40), "gray")),
                    "blue": ImageTk.PhotoImage(Image.new("RGB", (40, 40), "gray"))
                }

    def init_battlefield_for_setup(self):
        self.state_machine.transition_to(AppState.SETUP)
        self.battle_field = Battlefield(self.monster_data)
        self.units = []

        left_army_config = self.battle_data.get("left", {})
        right_army_config = self.battle_data.get("right", {})

        self.battle_field.setup_battle(
            left_army_config, right_army_config, self.monster_data)
        while self.battle_field.gameTime < 6.0:
            result = self.battle_field.run_one_frame()
            if result:
                break

    def _normalize_outcome(self, outcome):
        try:
            from simulator.utils import Faction
            if outcome == Faction.LEFT:
                return "LEFT"
            if outcome == Faction.RIGHT:
                return "RIGHT"
        except Exception:
            pass
        if outcome in ("LEFT", "RIGHT", "DRAW"):
            return outcome
        return "DRAW"

    def create_widgets(self):
        control_frame = tk.Frame(self.master)
        control_frame.pack(pady=5, fill=tk.X)

        top_control_frame = tk.Frame(control_frame)
        top_control_frame.pack(fill=tk.X)

        # 提示信息
        self.message_label = tk.Label(control_frame, text="", fg="black")
        self.message_label.pack(pady=2, anchor="w")

        # ====== 蒙特卡洛控制区 ======
        mc_frame = tk.LabelFrame(self.master, text="蒙特卡洛模拟", padx=10, pady=8)
        mc_frame.pack(fill=tk.X, padx=10, pady=8)

        tk.Label(mc_frame, text="模拟次数:").grid(row=0, column=0, sticky="w")
        self.mc_runs_entry = tk.Entry(mc_frame, width=10)
        self.mc_runs_entry.grid(row=0, column=1, sticky="w", padx=6)
        self.mc_runs_entry.insert(0, "1000")

        self.mc_seed_var = tk.StringVar(value="")  # 可选随机种子
        tk.Label(mc_frame, text="随机种子(可空):").grid(
            row=0, column=2, sticky="w", padx=(20, 0))
        self.mc_seed_entry = tk.Entry(
            mc_frame, width=12, textvariable=self.mc_seed_var)
        self.mc_seed_entry.grid(row=0, column=3, sticky="w", padx=6)

        self.mc_run_button = tk.Button(
            mc_frame, text="运行蒙卡", command=self.start_monte_carlo_threads)
        self.mc_run_button.grid(row=0, column=4, padx=12)

        self.mc_stop_button = tk.Button(
            mc_frame, text="停止", state=tk.DISABLED, command=self.stop_monte_carlo_threads)
        self.mc_stop_button.grid(row=0, column=5, padx=4)

        # ...
        self.mc_canvas = tk.Canvas(mc_frame, height=160, bg="white")
        self.mc_canvas.grid(row=2, column=0, columnspan=6,
                            sticky="we", pady=(6, 2))
        self.mc_canvas.bind(
            "<Configure>", lambda e: self._redraw_mc_canvas())  # 动态重绘

        # 结果文字
        self.mc_result_text = tk.Text(mc_frame, height=6, width=70)
        self.mc_result_text.grid(
            row=1, column=0, columnspan=5, pady=(8, 4), sticky="we")
        self.mc_result_text.configure(state=tk.DISABLED)

        # 简单条形图画布（显示胜负/平局分布）
        self.mc_canvas = tk.Canvas(mc_frame, height=160, bg="white")
        self.mc_canvas.grid(row=2, column=0, columnspan=5,
                            sticky="we", pady=(6, 2))
        self.canvas = None

    def run_single_battle(self, left_cfg, right_cfg):
        """基于给定左右双方配置，运行一局直到得出胜负；返回 ('LEFT'|'RIGHT'|'DRAW', 用时秒数)"""
        bf = Battlefield(self.monster_data)
        bf.setup_battle(left_cfg, right_cfg, self.monster_data)

        result = None
        # 安全上限，防止极端情况下无限跑（必要时可调大）
        max_steps = 20000000
        steps = 0
        while result is None and steps < max_steps:
            result = bf.run_one_frame()
            steps += 1

        # 结果为 Faction.LEFT / Faction.RIGHT / 其它(None or 特殊值)
        if result is None:
            outcome = "DRAW"
        else:
            try:
                from simulator.utils import Faction
                if result == Faction.LEFT:
                    outcome = "LEFT"
                elif result == Faction.RIGHT:
                    outcome = "RIGHT"
                else:
                    outcome = str(result)
            except Exception:
                outcome = str(result)
        return outcome, getattr(bf, "gameTime", 0.0)

    def _mc_task(self, n_times, left_cfg, right_cfg, seed_val, stop_event):
        if seed_val is not None:
            random.seed(seed_val)
        local = Counter()
        done = 0
        for _ in range(n_times):
            if stop_event.is_set():
                break
            outcome, _ = self.run_single_battle(left_cfg, right_cfg)
            local[self._normalize_outcome(outcome)] += 1
            done += 1
        # 返回 (Counter, 实际完成次数)
        return local, done

    def start_monte_carlo_threads(self):
        if self.mc_running:
            return

        # 读取参数
        try:
            runs = int(self.mc_runs_entry.get())
            if runs <= 0:
                raise ValueError
        except ValueError:
            self.show_message_below_button("蒙卡次数需为正整数", is_error=True)
            return

        seed_txt = self.mc_seed_var.get().strip()
        seed_val = None
        if seed_txt:
            try:
                seed_val = int(seed_txt)
            except ValueError:
                self.show_message_below_button("随机种子需为整数或留空", is_error=True)
                return

        left_cfg = copy.deepcopy(self.battle_data.get("left",  {}))
        right_cfg = copy.deepcopy(self.battle_data.get("right", {}))
        if not left_cfg or not right_cfg:
            self.show_message_below_button("错误：左右双方任一为空，无法进行蒙卡", is_error=True)
            return

        # 状态初始化
        with self.mc_lock:
            self._mc_counts.clear()
            self._mc_total_done = 0
            self._mc_total_target = runs
            self._mc_futures = []
            # 清空队列（防止上次遗留）
            while not self._mc_result_queue.empty():
                try:
                    self._mc_result_queue.get_nowait()
                except queue.Empty:
                    break

        self.mc_stop_event.clear()
        self._redraw_mc_canvas()

        # 线程池
        if self.executor is None:
            max_workers = min(8, max(1, (os.cpu_count() or 4) - 1))
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        workers = getattr(self.executor, "_max_workers", 4)

        # 关键：制造足够多的小任务（比如每个任务 10～50 次）
        # 这样就能频繁完成并推送结果，界面“按完成块”刷新
        TARGET_TASKS = min(max(32, workers * 16), runs)   # 至少几十个分片
        chunk = max(1, runs // TARGET_TASKS)

        remain = runs
        idx = 0
        while remain > 0:
            n = min(chunk, remain)
            fut = self.executor.submit(self._mc_task, n, left_cfg, right_cfg,
                                       None if seed_val is None else seed_val + idx,
                                       self.mc_stop_event)

            # 回调里**不要**动 UI；只把结果丢进队列
            def _on_done(f):
                try:
                    res = f.result()
                except Exception:
                    res = (Counter(), 0)
                # 线程安全：把 (Counter, done) 放进队列
                self._mc_result_queue.put(res)

            fut.add_done_callback(_on_done)
            self._mc_futures.append(fut)

            idx += 1
            remain -= n

        # UI 状态
        self.mc_run_button.config(state=tk.DISABLED)
        self.mc_stop_button.config(state=tk.NORMAL)
        self.mc_running = True

        # 定时消费队列 + 重画
        self.master.after(50, self._mc_tick_pool)  # 刷新更灵敏些

    def _mc_tick_pool(self):
        if not self.mc_running:
            return

        # 合并已完成任务的结果
        merged = False
        still_pending = []
        for fut in self._mc_futures:
            if fut.done():
                try:
                    local_counter, done = fut.result()
                except Exception as e:
                    local_counter, done = Counter(), 0  # 出错就忽略该分片
                with self.mc_lock:
                    self._mc_counts.update(local_counter)
                    self._mc_total_done += done
                merged = True
            else:
                still_pending.append(fut)
        self._mc_futures = still_pending

        # 渲染
        with self.mc_lock:
            counts = Counter(self._mc_counts)
            done = self._mc_total_done
            target = self._mc_total_target
        self._render_mc_text_and_chart(counts, done, target)

        # 完成或已请求停止
        if (done >= target) or self.mc_stop_event.is_set():
            # 不再等待未完成的分片（它们会尽快感知 stop_event 并返回较小 done）
            self._mc_finish()
            return

        # 继续轮询
        self.master.after(10, self._mc_tick_pool)

    def stop_monte_carlo_threads(self):
        if not self.mc_running:
            return
        self.mc_stop_event.set()

    def _mc_finish(self):
        # 最后一帧
        with self.mc_lock:
            counts = Counter(self._mc_counts)
            done = self._mc_total_done
            target = self._mc_total_target
        self._render_mc_text_and_chart(counts, done, target)

        self.mc_running = False
        self.mc_run_button.config(state=tk.NORMAL)
        self.mc_stop_button.config(state=tk.DISABLED)

        if done >= target and not self.mc_stop_event.is_set():
            self.show_message_below_button("蒙卡完成", is_error=False)
        else:
            self.show_message_below_button(
                f"已停止（完成 {done}/{target} 次）", is_error=False)
        self.mc_stop_event.clear()

    def _render_mc_text_and_chart(self, counts: Counter, done: int, target: int):
        left_win = counts.get("LEFT", 0)
        right_win = counts.get("RIGHT", 0)
        draw_cnt = counts.get("DRAW", 0)

        # 文本
        lines = []
        lines.append(f"已完成: {done}/{target}")
        if done > 0:
            lines.append(f"左方胜: {left_win} ({left_win / done:.2%})")
            lines.append(f"右方胜: {right_win} ({right_win / done:.2%})")
            lines.append(f"平局:   {draw_cnt} ({draw_cnt / done:.2%})")
        else:
            lines.append("左方胜: 0")
            lines.append("右方胜: 0")
            lines.append("平局:   0")
        self.mc_result_text.configure(state=tk.NORMAL)
        self.mc_result_text.delete("1.0", tk.END)
        self.mc_result_text.insert(tk.END, "\n".join(lines))
        self.mc_result_text.configure(state=tk.DISABLED)

        # 图
        items = [("左方胜", left_win), ("右方胜", right_win), ("平局", draw_cnt)]
        self.draw_mc_barchart(items, total=max(1, done))  # 避免除零

    def _redraw_mc_canvas(self):
        """在画布尺寸变化时重绘一帧当前统计"""
        with self.mc_lock:
            counts = Counter(self._mc_counts)
            done = self._mc_total_done
        items = [("左方胜", counts.get("LEFT", 0)),
                 ("右方胜", counts.get("RIGHT", 0)),
                 ("平局",   counts.get("DRAW", 0))]
        self.draw_mc_barchart(items, total=max(1, done))

    def draw_mc_barchart(self, items, total):
        """items: List[(label, count)]"""
        if not self.mc_canvas:
            return
        c = self.mc_canvas
        c.delete("all")
        if not items or total <= 0:
            c.create_text(10, 10, text="无数据", anchor="nw")
            return

        W = int(c.winfo_width()) or 600
        H = int(c.winfo_height()) or 160
        pad_x = 40
        pad_y = 24
        bar_space = (W - 2*pad_x)
        bar_w = bar_space // max(1, len(items)) - 16
        max_cnt = max(x[1] for x in items) or 1

        # 坐标轴
        c.create_line(pad_x, H - pad_y, W - pad_x//2, H - pad_y)  # x轴
        c.create_line(pad_x, pad_y//2, pad_x, H - pad_y)          # y轴

        for i, (label, cnt) in enumerate(items):
            x0 = pad_x + i * (bar_w + 16) + 8
            x1 = x0 + bar_w
            # 高度按计数线性映射
            h_ratio = cnt / max_cnt if max_cnt > 0 else 0.0
            bar_h = int((H - 2*pad_y) * h_ratio)
            y0 = H - pad_y - bar_h
            y1 = H - pad_y
            # 绘制条
            c.create_rectangle(x0, y0, x1, y1, fill="#6aa84f", outline="")
            # 文本：计数与百分比
            pct = f"{(cnt/total):.1%}"
            c.create_text((x0+x1)//2, y0-10, text=f"{cnt} ({pct})", anchor="s")
            # 类别名
            c.create_text((x0+x1)//2, H - pad_y + 12, text=label, anchor="n")

    def refresh_canvas_display(self):
        if self.monte_carlo_mode:
            return
        # ... 原本的画网格/单位/连线/计时 等逻辑保持不变

    def show_message_below_button(self, message, is_error=False, duration=5000):
        """在按钮下方显示文字提示，并在指定时间后消失"""
        if self.message_label:
            self.message_label.config(
                text=message, fg="red" if is_error else "black")
            if self.message_timer_id:
                self.master.after_cancel(self.message_timer_id)
            self.message_timer_id = self.master.after(
                duration, self.clear_message_below_button)

    def clear_message_below_button(self):
        """清除提示信息"""
        if self.message_label:
            self.message_label.config(text="")
        self.message_timer_id = None

    def enter_setup_phase(self):
        if self.state_machine.state in [AppState.SIMULATING, AppState.PAUSED]:
            # 简单起见，直接中断，不弹出确认框
            pass
        print("进入部署阶段。使用 self.battle_data 初始化战场。")
        self.state_machine.transition_to(AppState.SETUP)
        self.init_battlefield_for_setup()  # 这会调用 refresh_canvas_display，进而清除旧的胜利信息


def main():
    root = tk.Tk()
    # root.withdraw() # 如果不需要立即隐藏主窗口，可以注释掉

    initial_battle_setup = {"left": {"炮击组长": 3, "“庞贝”": 2}, "right": {"沸血骑士团精锐": 6, "炽焰源石虫": 23},
                            "result": "left"}

    sys.stdin.reconfigure(encoding='utf-8')
    try:
        if not sys.stdin.isatty():
            json_data = sys.stdin.read()
            if json_data:
                initial_battle_setup = json.loads(json_data)
                print("成功从 stdin 加载战斗配置。")
            else:
                print("stdin 中没有数据，使用默认配置。")
        else:
            print("stdin 是交互式终端，使用默认配置。")
    except json.JSONDecodeError:
        print("错误: 无法解析 stdin 中的 JSON 数据，请检查格式，将使用默认配置。")

    except Exception as e:
        print(f"从 stdin 读取或解析时发生未知错误: {e}，将使用默认配置。")

    app = SandboxSimulator(root, initial_battle_setup)

    error_messages = []
    if not app.monster_data:  # 检查 monsters.json 是否加载成功
        error_messages.append("错误: monsters.json 未找到或为空，请检查文件！")

    if len(initial_battle_setup.get("left", {})) == 0 or len(initial_battle_setup.get("right", {})) == 0:
        error_messages.append("错误：至少一方的怪物列表为空！")

    problematic_monsters_found = []
    problematic_monster_names = ["矿脉守卫", "提亚卡乌好战者", "凋零萨卡兹", "狂暴宿主组长", "高能源石虫"]

    for team_key in ["left", "right"]:
        for monster_name in initial_battle_setup.get(team_key, {}):
            if monster_name in problematic_monster_names:
                problematic_monsters_found.append(
                    f"{('左方' if team_key == 'left' else '右方')}存在问题怪物: {monster_name}")

    if problematic_monsters_found:
        error_messages.append("警告: " + "； ".join(problematic_monsters_found))

    if error_messages:
        app.show_message_below_button(" | ".join(
            error_messages), is_error=True, duration=10000)
        if "monsters.json 未找到或为空" in " | ".join(error_messages):
            print("由于 monsters.json 缺失或错误，模拟器可能无法正常工作。")

    if not root.winfo_exists():
        return
    app.master.deiconify()  # 显示主窗口
    root.mainloop()


if __name__ == "__main__":
    main()
