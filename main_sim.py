import copy
from enum import Enum, auto
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import math
from unit import Unit  # 确保 Unit 已导入
import json  # REMOVED_TEAM_INTERFACE: Added missing import for the main block
import random  # REMOVED_TEAM_INTERFACE: Added missing import for the main block
import sys  # Import sys for stdin
import logging

from simulator.battle_field import Battlefield  # 确保 Battlefield 已导入
from simulator.monsters import MonsterFactory  # 确保 MonsterFactory 已导入
from simulator.utils import MONSTER_MAPPING, REVERSE_MONSTER_MAPPING, Faction  # 根据你的实际路径调整
from simulator.vector2d import FastVector  # 确保 FastVector 已导入
from simulator.monsters import AttackState, Monster, MonsterFactory
from recognize import MONSTER_COUNT

logger = logging.getLogger(__name__)

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
            logger.error(f"Illegal state transition from {self.state} to {new_state}")

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

        self.simulating = False  # 战斗模拟是否进行中
        self.simulation_id = None
        self.speed_multiplier = 10  # 默认速度
        self.is_paused = False  # 新增：暂停状态

        self.monster_data = []  # 在 load_assets 中加载
        self.load_assets()  # 先加载资源，特别是 self.monster_data

        self.battle_field = Battlefield(self.monster_data)  # 核心战场逻辑对象
        self.initial_battlefield = None
        # 新增状态
        self.setup_phase_active = False  # 怪物部署和调整阶段是否激活
        self.selected_monster_for_drag = None  # 当前拖动的怪物对象 (Monster 类型)

        self.message_label = None  # 用于显示提示信息的标签
        self.message_timer_id = None  # 用于定时清除提示信息的ID

        self.create_widgets()
        self.master.protocol("WM_DELETE_WINDOW", self.hide_window)
        self.state_machine = StateMachine(self.update_ui_state)
        self.state_machine.transition_to(AppState.INITIAL)
        self.enter_setup_phase()

    def update_ui_state(self):
        """根据当前状态更新所有控件状态"""
        states = self.state_machine.get_control_states()

        self.deploy_button.config(
            state=states['deploy']['state'],
            text=states['deploy']['text']
        )
        self.confirm_start_button.config(state=states['confirm_start']['state'])
        self.pause_button.config(
            state=states['pause']['state'],
            text=states['pause']['text']
        )
        self.restore_button.config(state=states['restore']['state'])
        self.speed_entry.config(state=states['speed_entry']['state'])
        self.clear_button.config(state=states['clear']['state'])

        if states['timer']['text'] != '':
            self.timer_label.config(text=states['timer']['text'])

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
            logger.error("错误: monsters.json 未找到，请检查路径！")
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
                logger.error(f"加载图标错误 (图标键: {i}, 文件名ID: {image_file_id}): {str(e)}")
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

        self.battle_field.setup_battle(left_army_config, right_army_config, self.monster_data)
        while self.battle_field.gameTime < 6.0:
            result = self.battle_field.run_one_frame()
            if result:
                break
        self.refresh_canvas_display()

    def on_mouse_drag(self, event):
        if self.selected_monster_for_drag and self.state_machine.state in [AppState.PAUSED, AppState.SETUP]:
            new_grid_x = event.x / self.cell_size
            new_grid_y = event.y / self.cell_size
            new_grid_x = max(0.25, min(new_grid_x, self.grid_width - 0.25))
            new_grid_y = max(0.25, min(new_grid_y, self.grid_height - 0.25))

            self.selected_monster_for_drag.position.x = new_grid_x
            self.selected_monster_for_drag.position.y = new_grid_y
            self.selected_monster_for_drag.target_deployment_position = FastVector(new_grid_x, new_grid_y)

            self.refresh_canvas_display()

    def create_widgets(self):
        control_frame = tk.Frame(self.master)
        control_frame.pack(pady=5)

        top_control_frame = tk.Frame(control_frame)
        top_control_frame.pack(fill=tk.X)

        speed_frame = tk.Frame(top_control_frame)
        speed_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(speed_frame, text="倍速:").pack(side=tk.LEFT)
        self.speed_entry = tk.Entry(speed_frame, width=5)
        self.speed_entry.pack(side=tk.LEFT)
        self.speed_entry.insert(0, f"{int(self.speed_multiplier)}")
        tk.Button(speed_frame, text="应用", command=self.apply_speed).pack(side=tk.LEFT)

        self.deploy_button = tk.Button(top_control_frame, text="部署怪物", command=self.enter_setup_phase)
        self.deploy_button.pack(side=tk.LEFT, padx=5)

        self.confirm_start_button = tk.Button(top_control_frame, text="开始战斗", command=self.start_actual_simulation,
                                              state=tk.DISABLED)
        self.confirm_start_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(top_control_frame, text="清空战场", command=self.clear_sandbox)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        self.timer_label = tk.Label(top_control_frame, text="0.00秒")
        self.timer_label.pack(side=tk.LEFT, padx=100)

        self.pause_button = tk.Button(top_control_frame, text="暂停", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)

        self.restore_button = tk.Button(top_control_frame, text="恢复站位", command=self.restore_initial_positions,
                                        state=tk.DISABLED)
        self.restore_button.pack(side=tk.LEFT, padx=5)

        self.game_over_label = tk.Label(top_control_frame, text="", fg="red")
        self.game_over_label.pack(side=tk.LEFT, padx=5)

        # 用于显示提示信息的标签，放置在按钮行的下方
        self.message_label = tk.Label(control_frame, text="", fg="black")
        self.message_label.pack(pady=2)

        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack(pady=10)
        self.draw_grid()

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def show_message_below_button(self, message, is_error=False, duration=5000):
        """在按钮下方显示文字提示，并在指定时间后消失"""
        if self.message_label:
            self.message_label.config(text=message, fg="red" if is_error else "black")
            if self.message_timer_id:
                self.master.after_cancel(self.message_timer_id)
            self.message_timer_id = self.master.after(duration, self.clear_message_below_button)

    def clear_message_below_button(self):
        """清除提示信息"""
        if self.message_label:
            self.message_label.config(text="")
        self.message_timer_id = None

    def toggle_pause(self):
        if self.state_machine.state == AppState.SIMULATING:
            self.state_machine.transition_to(AppState.PAUSED)
        elif self.state_machine.state == AppState.PAUSED:
            self.state_machine.transition_to(AppState.SIMULATING)
            self.simulate()

    def apply_speed(self):
        try:
            new_speed = float(self.speed_entry.get())
            if new_speed <= 0:
                raise ValueError("速度必须大于0")
            self.speed_multiplier = new_speed
        except ValueError as e:
            self.show_message_below_button(f"无效的速度值: {e}", is_error=True)
            self.speed_entry.delete(0, tk.END)
            self.speed_entry.insert(0, str(self.speed_multiplier))

    def draw_grid(self):
        danger_zone = 0
        if self.battle_field and self.battle_field.danger_zone_size() > 0:
            danger_zone = min(self.battle_field.danger_zone_size(), self.grid_height / 2 + 1)
            self.canvas.create_rectangle(0, 0, self.canvas_width, danger_zone * self.cell_size, fill='#cccc00',
                                         outline="")
            self.canvas.create_rectangle(0, 0, (danger_zone+1) * self.cell_size , self.canvas_height, fill='#cccc00',
                                         outline="")
            self.canvas.create_rectangle(self.canvas_width, 0, self.canvas_width - (danger_zone+1) * self.cell_size,
                                         self.canvas_height, fill='#cccc00', outline="")
            self.canvas.create_rectangle(self.canvas_width, self.canvas_height, 0,
                                         self.canvas_height - danger_zone * self.cell_size, fill='#cccc00', outline="")

        # 绘制最左侧的柔和红色列
        soft_red_fill = '#F08080'  # 亮珊瑚色
        soft_red_line = '#D87070'  # 稍暗的亮珊瑚色
        self.canvas.create_rectangle(0, 0, self.cell_size, self.canvas_height, fill=soft_red_fill, outline="")

        # 绘制最右侧的柔和蓝色列
        soft_blue_fill = '#ADD8E6'  # 浅蓝色
        soft_blue_line = '#9CC2D0'  # 稍暗的浅蓝色
        self.canvas.create_rectangle(self.canvas_width - self.cell_size, 0, self.canvas_width, self.canvas_height,
                                     fill=soft_blue_fill, outline="")

        for i in range(self.grid_width + 1):
            x = i * self.cell_size
            if i == 0 or i == self.grid_width:
                continue
            if i == 1:  # 红色列的右边缘线
                self.canvas.create_line(x, 0, x, self.canvas_height, fill=soft_red_line)
            elif i == self.grid_width - 1:  # 蓝色列的左边缘线
                self.canvas.create_line(x, 0, x, self.canvas_height, fill=soft_blue_line)
            else:
                self.canvas.create_line(x, 0, x, self.canvas_height, fill='lightgray')  # 垂直网格线

        # 绘制水平网格线
        for i in range(self.grid_height + 1):
            y = i * self.cell_size
            # 左边红色区域的水平线可以特殊处理，使其在红色背景上更明显，或者保持lightgray
            if self.cell_size > 0:  # 确保 cell_size > 0 避免问题
                # 在红色区域内绘制颜色稍浅的水平线，或者使用对比色
                self.canvas.create_line(0, y, self.cell_size, y, fill='#E07070')  # 示例：红色区域内的水平线
                # 在蓝色区域内绘制颜色稍浅的水平线
                self.canvas.create_line(self.canvas_width - self.cell_size, y, self.canvas_width, y,
                                        fill='#9CBED0')  # 示例：蓝色区域内的水平线
                # 中间区域的水平线
                self.canvas.create_line(self.cell_size, y, self.canvas_width - self.cell_size, y, fill='lightgray')
            else:  # 如果 cell_size 为0或负，则绘制完整线条
                self.canvas.create_line(0, y, self.canvas_width, y, fill='lightgray')

    def refresh_canvas_display(self):
        if not self.canvas: return
        self.canvas.delete("all")
        self.draw_grid()
        self.canvas.delete("victory_text")  # 清除可能存在的胜利信息

        if len(self.battle_field.monsters) > len(self.units):
            for i in range(len(self.units), len(self.battle_field.monsters)):
                new_unit = Unit('red', 0, 0, 0)
                self.units.append(new_unit)
        elif len(self.battle_field.monsters) < len(self.units):
            self.units = self.units[:len(self.battle_field.monsters)]

        index = 0
        for monster in self.battle_field.monsters:
            ui_unit = self.units[index]
            index += 1
            if monster and monster.is_alive:
                ui_unit.x = monster.position.x
                ui_unit.y = monster.position.y
                ui_unit.team = 'red' if monster.faction == Faction.LEFT else 'blue'
                display_id_for_icon = REVERSE_MONSTER_MAPPING.get(monster.name)
                if display_id_for_icon is None:
                    logger.error(f"怪物名 {monster.name} 在 REVERSE_MONSTER_MAPPING 中未找到!")
                    self.show_message_below_button(f"怪物名 {monster.name} 在 REVERSE_MONSTER_MAPPING 中未找到!",
                                                   is_error=True)
                    display_id_for_icon = 0
                ui_unit.unit_id = display_id_for_icon
                ui_unit.health = monster.health
                ui_unit.max_health = monster.max_health
                ui_unit.skill = monster.get_skill_bar()
                ui_unit.monster_global_id = monster.id
                ui_unit.max_skill = monster.get_max_skill_bar()
                self.draw_unit(ui_unit, monster)

        if self.state_machine.state in [AppState.SIMULATING, AppState.PAUSED]:
            for monster_obj in self.battle_field.monsters:
                if monster_obj.is_alive and monster_obj.target is not None:
                    self.canvas.create_line(
                        monster_obj.position.x * self.cell_size, monster_obj.position.y * self.cell_size,
                        monster_obj.target.position.x * self.cell_size, monster_obj.target.position.y * self.cell_size,
                        fill="#FF3030" if monster_obj.faction == Faction.LEFT else "#3030FF", width=1, arrow='last')
            self.timer_label.config(text=f"{self.battle_field.gameTime:.2f}秒")

    def draw_unit(self, unit, monster: 'Monster'):
        x_pixel = unit.x * self.cell_size
        y_pixel = unit.y * self.cell_size

        icon_to_draw = None
        if unit.unit_id in self.icons and unit.team in self.icons[unit.unit_id]:
            icon_to_draw = self.icons[unit.unit_id][unit.team]
        else:
            self.canvas.create_rectangle(x_pixel - 20, y_pixel - 20, x_pixel + 20, y_pixel + 20, fill="gray",
                                         tags=("unit",))
            return

        self.canvas.create_image(x_pixel, y_pixel, image=icon_to_draw, tags=("unit",))

        bar_width = 40
        bar_height = 5
        health_bar_y = y_pixel + 25
        health_ratio = min(1, unit.health / unit.max_health) if unit.max_health > 0 else 0
        current_health_width = max(0, bar_width * health_ratio)
        self.canvas.create_rectangle(x_pixel - bar_width / 2, health_bar_y - bar_height / 2, x_pixel + bar_width / 2,
                                     health_bar_y + bar_height / 2, fill="#400000", outline="")
        if current_health_width > 0:
            self.canvas.create_rectangle(x_pixel - bar_width / 2, health_bar_y - bar_height / 2,
                                         x_pixel - bar_width / 2 + current_health_width, health_bar_y + bar_height / 2,
                                         fill="#FF3030", outline="")

        skill_bar_y = health_bar_y + 7
        skill_ratio = min(1, unit.skill / unit.max_skill) if unit.max_skill > 0 else 0
        current_skill_width = bar_width * skill_ratio
        self.canvas.create_rectangle(x_pixel - bar_width / 2, skill_bar_y - bar_height / 2, x_pixel + bar_width / 2,
                                     skill_bar_y + bar_height / 2, fill="black", outline="")

        attack_state_color = "#ffcccc"
        if monster.attack_state == AttackState.后摇:
            attack_state_color = "#30FF30"
        elif monster.attack_state == AttackState.等待:
            attack_state_color = "yellow"

        if current_skill_width > 0:
            self.canvas.create_rectangle(x_pixel - bar_width / 2, skill_bar_y - bar_height / 2,
                                         x_pixel - bar_width / 2 + current_skill_width, skill_bar_y + bar_height / 2,
                                         fill=attack_state_color, outline="")

    def on_mouse_down(self, event):
        grid_x = event.x / self.cell_size
        grid_y = event.y / self.cell_size

        if self.state_machine.state in [AppState.PAUSED, AppState.SETUP]:
            clicked_monster_obj = None
            for unit_in_list in reversed(self.units):
                monster = self.battle_field.get_monster_with_id(unit_in_list.monster_global_id)
                if not monster: continue
                monster_center_x_px = monster.position.x * self.cell_size
                monster_center_y_px = monster.position.y * self.cell_size
                icon_half_size_px = 20
                if (abs(event.x - monster_center_x_px) < icon_half_size_px and
                        abs(event.y - monster_center_y_px) < icon_half_size_px):
                    clicked_monster_obj = monster
                    break
            if clicked_monster_obj:
                self.selected_monster_for_drag = clicked_monster_obj
                return

    def on_mouse_drag(self, event):
        if self.selected_monster_for_drag and self.state_machine.state in [AppState.PAUSED, AppState.SETUP]:
            new_grid_x = event.x / self.cell_size
            new_grid_y = event.y / self.cell_size
            new_grid_x = max(0.25, min(new_grid_x, self.grid_width - 0.25))
            new_grid_y = max(0.25, min(new_grid_y, self.grid_height - 0.25))
            self.selected_monster_for_drag.position.x = new_grid_x
            self.selected_monster_for_drag.position.y = new_grid_y
            self.selected_monster_for_drag.target = self.selected_monster_for_drag.find_target()
            self.refresh_canvas_display()

    def on_mouse_up(self, event):
        if self.selected_monster_for_drag:
            pass
        for monster in self.battle_field.alive_monsters:
            monster.target = monster.find_target()
        self.selected_monster_for_drag = None

    def enter_setup_phase(self):
        if self.state_machine.state in [AppState.SIMULATING, AppState.PAUSED]:
            # 简单起见，直接中断，不弹出确认框
            pass
        self.game_over_label.config(text="")
        logger.info("进入部署阶段。使用 self.battle_data 初始化战场。")
        self.state_machine.transition_to(AppState.SETUP)
        self.init_battlefield_for_setup()  # 这会调用 refresh_canvas_display，进而清除旧的胜利信息

    def start_actual_simulation(self):
        if self.state_machine.state != AppState.SETUP:
            self.show_message_below_button("请先通过'部署怪物'加载初始设置。")
            return
        if not self.battle_field.monsters:
            self.show_message_below_button("战场上没有怪物，无法开始模拟。")
            return
        self.initial_battlefield = copy.deepcopy(self.battle_field)
        logger.info("战斗模拟开始！")
        self.state_machine.transition_to(AppState.SIMULATING)
        self.game_over_label.config(text="")
        self.canvas.delete("victory_text")  # 开始战斗前清除旧的胜利信息
        self.simulate()

    def simulate(self):
        if self.state_machine.state != AppState.SIMULATING:
            return
        result = None
        for _ in range(int(self.speed_multiplier)):
            result = self.battle_field.run_one_frame()
            if result is not None:
                break

        self.refresh_canvas_display()  # 先刷新单位位置

        if result is not None:
            self.state_machine.transition_to(AppState.ENDED)
            winner_faction = result
            game_over_message = ""
            text_color = "black"
            font_style = ("Arial", 40, "bold")  # 统一字体

            if winner_faction == Faction.LEFT:
                game_over_message = "左方胜利！"
                text_color = "red"
            elif winner_faction == Faction.RIGHT:
                game_over_message = "右方胜利！"
                text_color = "blue"
            else:
                game_over_message = f"游戏结束，结果: {result}"

            # 在画布中央放大显示信息
            self.canvas.create_text(
                self.canvas_width / 2, self.canvas_height / 2,
                text=game_over_message, font=font_style, fill=text_color, tags="victory_text"
            )
        else:
            interval = max(1, 33)
            self.simulation_id = self.master.after(interval, self.simulate)

    def show_result(self, message):  # 此方法现在主要被画布显示和按钮下方提示替代
        self.show_message_below_button(message)

    def clear_sandbox(self):
        if self.state_machine.state == AppState.SIMULATING:
            self.master.after_cancel(self.simulation_id)

        self.state_machine.transition_to(AppState.INITIAL)
        self.units = []
        self.battle_field = Battlefield(self.monster_data)
        self.selected_monster_for_drag = None
        self.selected_team = None
        self.selected_unit_id = None
        if self.canvas:
            self.refresh_canvas_display()  # 会调用 draw_grid 并清除 "all" 包括 "victory_text"
        self.game_over_label.config(text="")
        self.clear_message_below_button()
        logger.info("战场已清空。")

    def restore_initial_positions(self):
        if not self.initial_battlefield:
            self.show_message_below_button("没有可恢复的初始站位")
            return
        self.battle_field = copy.deepcopy(self.initial_battlefield)
        self.battle_field.gameTime = self.initial_battlefield.gameTime  # 保持时间连续性
        if self.state_machine.state in [AppState.PAUSED, AppState.ENDED]:
            self.state_machine.transition_to(AppState.SETUP)
        self.refresh_canvas_display()  # 会清除 "victory_text"
        logger.info("已恢复到初始站位。")


def main():
    root = tk.Tk()
    # root.withdraw() # 如果不需要立即隐藏主窗口，可以注释掉

    initial_battle_setup = {"left": {"Vvan": 4, "炮击组长": 3, "“庞贝”": 2}, "right": {"大喷蛛": 6, "冰爆源石虫": 23},
                            "result": "left"}

    sys.stdin.reconfigure(encoding='utf-8')
    try:
        if not sys.stdin.isatty():
            json_data = sys.stdin.read()
            if json_data:
                initial_battle_setup = json.loads(json_data)
                logger.info("成功从 stdin 加载战斗配置。")
            else:
                logger.error("stdin 中没有数据，使用默认配置。")
        else:
            logger.info("stdin 是交互式终端，使用默认配置。")
    except json.JSONDecodeError:
        logger.error("错误: 无法解析 stdin 中的 JSON 数据，请检查格式，将使用默认配置。")

    except Exception as e:
        logger.error(f"从 stdin 读取或解析时发生未知错误: {e}，将使用默认配置。")

    try:
        app = SandboxSimulator(root, initial_battle_setup)
    except Exception as e:
        logger.error(f"初始化模拟器时发生错误: {e}")
        messagebox.showerror("初始化错误", f"无法初始化模拟器: {e}")
        root.destroy()
        return

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
        app.show_message_below_button(" | ".join(error_messages), is_error=True, duration=10000)
        if "monsters.json 未找到或为空" in " | ".join(error_messages):
            logger.error("由于 monsters.json 缺失或错误，模拟器可能无法正常工作。")


    if not root.winfo_exists():
        return
    app.master.deiconify()  # 显示主窗口
    root.mainloop()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("PIL").setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    ))
    logging.getLogger().addHandler(stream_handler)
    main()
