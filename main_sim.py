import copy
from enum import Enum, auto
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import math
from simulator.battle_field import Battlefield  # 确保 Battlefield 已导入
from simulator.monsters import MonsterFactory  # 确保 MonsterFactory 已导入
from simulator.utils import MONSTER_MAPPING, REVERSE_MONSTER_MAPPING, Faction  # 根据你的实际路径调整
from simulator.vector2d import FastVector  # 确保 FastVector 已导入
from unit import Unit  # 确保 Unit 已导入
import json  # REMOVED_TEAM_INTERFACE: Added missing import for the main block
import random  # REMOVED_TEAM_INTERFACE: Added missing import for the main block
import sys # Import sys for stdin
from simulator.monsters import AttackState, Monster, MonsterFactory

class AppState(Enum):
    INITIAL = auto()        # 初始状态
    SETUP = auto()          # 部署阶段
    SIMULATING = auto()     # 模拟运行中
    PAUSED = auto()         # 暂停状态
    ENDED = auto()          # 战斗结束

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
        self.master = tk.Toplevel(master)
        self.master.title("沙盒模拟器")
        self.num_monsters = 58  # 根据你的怪物总数调整
        self.load_assets()
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
        self.is_paused = False   # 新增：暂停状态

        self.battle_field = Battlefield(self.monster_data)  # 核心战场逻辑对象
        self.initial_battlefield = None
        # 新增状态
        self.setup_phase_active = False  # 怪物部署和调整阶段是否激活
        self.selected_monster_for_drag = None  # 当前拖动的怪物对象 (Monster 类型)
        # self.dragging_offset = {'x': 0, 'y': 0} # 如果需要精确点击点拖动

        self.create_widgets()
        self.master.protocol("WM_DELETE_WINDOW", self.hide_window)
        self.state_machine = StateMachine(self.update_ui_state)
        self.state_machine.transition_to(AppState.INITIAL)        
        self.enter_setup_phase()

    def update_ui_state(self):
        """根据当前状态更新所有控件状态"""
        states = self.state_machine.get_control_states()
        
        # 更新按钮状态
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
        if self.state_machine.state == AppState.SIMULATING:  # 如果正在模拟，先停止
            self.master.after_cancel(self.simulation_id)
            self.state_machine.transition_to(AppState.PAUSED)
        self.master.withdraw()

    def load_assets(self):
        self.icons = {}
        try:
            with open("simulator/monsters.json", encoding='utf-8') as f:
                self.monster_data = json.load(f)["monsters"]
        except FileNotFoundError:
            messagebox.showerror("错误", "monsters.json 未找到，请检查路径！")
            self.monster_data = []  # 提供一个空列表以避免后续错误
            return  # 无法加载资源，可能需要更强的错误处理

        # 修改开始
        for i in range(self.num_monsters):  # i 将会从 0 遍历到 num_monsters - 1
            image_file_id = i + 1  # 图片文件名仍然是 1.png, 2.png ...
            try:
                image = Image.open(f'images/{image_file_id}.png') # 加载对应的图片文件
                self.icons[i] = {  # <--- 修改点：使用 i (0-indexed) 作为 self.icons 的键
                    "red": ImageTk.PhotoImage(image.resize((40, 40))),
                    "blue": ImageTk.PhotoImage(image.resize((40, 40)).transpose(Image.FLIP_LEFT_RIGHT))
                }
            except Exception as e:
                # 确保错误处理中也使用 i 作为键
                messagebox.showerror(f"加载图标错误 (图标键: {i}, 文件名ID: {image_file_id})", str(e))
                self.icons[i] = { # <--- 修改点：使用 i (0-indexed) 作为 self.icons 的键
                    "red": ImageTk.PhotoImage(Image.new("RGB", (40, 40), "gray")),
                    "blue": ImageTk.PhotoImage(Image.new("RGB", (40, 40), "gray"))
                }
        # 修改结束

    def init_battlefield_for_setup(self):
        """
        初始化战场用于部署阶段：
        1. 重置 Battlefield 对象。
        2. 根据 self.battle_data 生成怪物到战场 (self.battle_field.monsters)，
           此时怪物的位置是玩家预设的最终位置，但它们还不会移动。
        3. 同步 self.units 列表。
        """
        self.state_machine.transition_to(AppState.SETUP)
        self.battle_field = Battlefield(self.monster_data)
        self.units = []

        left_army_config = self.battle_data.get("left", {})
        right_army_config = self.battle_data.get("right", {})

        self.battle_field.setup_battle(left_army_config, right_army_config, self.monster_data)
        while self.battle_field.gameTime < 5.0:
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

            # 更新怪物核心逻辑对象的位置
            self.selected_monster_for_drag.position.x = new_grid_x
            self.selected_monster_for_drag.position.y = new_grid_y
            # 同时更新其目标部署位置，因为拖拽就是为了设定这个
            self.selected_monster_for_drag.target_deployment_position = FastVector(new_grid_x, new_grid_y)

            self.refresh_canvas_display()

    # def start_actual_simulation(self):
    #     if not self.state_machine.state in [AppState.SETUP]: 
    #         messagebox.showinfo("提示", "请先通过'部署怪物'加载并调整初始设置。")
    #         return
    #         # if not any(m.initial_spawn_position_set for m in self.battle_field.monsters):  # 如果没有任何怪物设置过目标部署点
                
    #         #     return
    #     if not self.battle_field.monsters:
    #         messagebox.showinfo("提示", "战场上没有怪物，无法开始模拟。")
    #         return

    #     self.state_machine.transition_to(AppState.SIMULATING)
    #     # self.setup_phase_active = False  # 退出部署阶段
    #     # self.simulating = True
    #     # self.confirm_start_button.config(state=tk.DISABLED)
    #     # self.deploy_button.config(text="部署怪物")  # 或者 "停止模拟"
    #     self.battle_field.gameTime = 0
    #     self.battle_field.round = 0
    #     # ... (其他清理)

    #     # 为每个怪物设置入场点并启动部署移动
    #     for monster in self.battle_field.monsters:
    #         if not monster.is_alive: continue  # 跳过已死亡（理论上不应该）或未正确初始化的

    #         target_deploy_pos = FastVector(monster.target_deployment_position.x,
    #                                        monster.target_deployment_position.y)  # 获取之前设定的目标位置

    #         # 定义战场边缘的出生点
    #         spawn_y = monster.target_deployment_position.y  # Y轴通常保持和目标一致，或者也可以随机一点
    #         if monster.faction == Faction.LEFT:
    #             initial_spawn_x = 0.1  # 从左侧边缘进入
    #         else:  # Faction.RIGHT
    #             initial_spawn_x = self.grid_width - 0.1  # 从右侧边缘进入

    #         initial_spawn_pos = FastVector(initial_spawn_x, spawn_y)

    #         monster.start_deployment(initial_spawn_pos, target_deploy_pos)
    #         print(f"启动部署: {monster.name}{monster.id} 从 {initial_spawn_pos} 前往 {target_deploy_pos}")

    #     print("战斗模拟开始！怪物正在进入战场...")
    #     self.simulate()  # 开始模拟循环

    # ... (simulate, refresh_canvas_display等其他方法)

    # def refresh_canvas_display(self):
    #     if not self.canvas: return
    #     self.canvas.delete("all")
    #     self.draw_grid()

    #     # temp_units_to_render = [] # 这行似乎没用了，可以直接遍历 self.units
    #     for ui_unit in self.units:
    #         monster = self.battle_field.get_monster_with_id(ui_unit.monster_global_id)
    #         if monster:
    #             # 即使在部署移动中，UI也应该跟随怪物的实际位置 (monster.position)
    #             ui_unit.x = monster.position.x
    #             ui_unit.y = monster.position.y
    #             ui_unit.team = 'red' if monster.faction == Faction.LEFT else 'blue'
    #             # ... (其余的UI更新逻辑不变)
    #             display_id_for_icon = REVERSE_MONSTER_MAPPING.get(monster.name)
    #             if display_id_for_icon is None:
    #                 # messagebox.showerror(f"错误", f"怪物名 {monster.name} 在 REVERSE_MONSTER_MAPPING 中未找到!")
    #                 print(f"警告: 怪物名 {monster.name} 在 REVERSE_MONSTER_MAPPING 中未找到! 使用默认图标。")
    #                 display_id_for_icon = 0  # 或者其他默认值
    #             ui_unit.unit_id = display_id_for_icon

    #             # 在部署阶段或模拟未开始时，血条显示满血
    #             # 在入场移动 (is_deploying) 时，也应该显示满血，因为它们还没开始战斗
    #             if self.setup_phase_active or not self.simulating or monster.is_deploying:
    #                 ui_unit.health = monster.max_health
    #             else:
    #                 ui_unit.health = monster.health
    #             ui_unit.max_health = monster.max_health

    #             # 技能条/攻击CD条在部署和入场阶段可以考虑显示为0或不显示特定状态
    #             if monster.is_deploying:
    #                 ui_unit.skill = 0  # 或者根据你的偏好调整
    #             else:
    #                 ui_unit.skill = monster.get_skill_bar()
    #             ui_unit.max_skill = monster.get_max_skill_bar()

    #             self.draw_unit(ui_unit, monster)  # monster对象可以传递更多状态给draw_unit
    #             # temp_units_to_render.append(ui_unit) # 如果上面注释了，这里也注释

    #     # ... (绘制攻击连线和计时器逻辑)
    #     if self.simulating:
    #         # 只有在非部署阶段的怪物才绘制攻击连线
    #         for monster_obj in self.battle_field.alive_monsters:
    #             if monster_obj.is_alive and not monster_obj.is_deploying and monster_obj.target is not None:
    #                 self.canvas.create_line(
    #                     monster_obj.position.x * self.cell_size, monster_obj.position.y * self.cell_size,
    #                     monster_obj.target.position.x * self.cell_size, monster_obj.target.position.y * self.cell_size,
    #                     fill="#FF3030" if monster_obj.faction == Faction.LEFT else "#3030FF", width=1, arrow='last')
    #         self.timer_label.config(text=f"{self.battle_field.gameTime:.2f}秒")

    def create_widgets(self):
        control_frame = tk.Frame(self.master)
        control_frame.pack(pady=5)

        speed_frame = tk.Frame(control_frame)
        speed_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(speed_frame, text="倍速:").pack(side=tk.LEFT)
        self.speed_entry = tk.Entry(speed_frame, width=5)
        self.speed_entry.pack(side=tk.LEFT)
        self.speed_entry.insert(0, f"{int(self.speed_multiplier)}")
        tk.Button(speed_frame, text="应用", command=self.apply_speed).pack(side=tk.LEFT)

        self.deploy_button = tk.Button(control_frame, text="部署怪物", command=self.enter_setup_phase)
        self.deploy_button.pack(side=tk.LEFT, padx=5)

        self.confirm_start_button = tk.Button(control_frame, text="开始战斗", command=self.start_actual_simulation,
                                              state=tk.DISABLED)
        self.confirm_start_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(control_frame, text="清空战场", command=self.clear_sandbox)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        self.timer_label = tk.Label(control_frame, text="0.00秒")
        self.timer_label.pack(side=tk.LEFT, padx=100)  # REMOVED_TEAM_INTERFACE: Adjusted to side=tk.LEFT to fill space, or use fill=tk.X

        self.pause_button = tk.Button(control_frame, text="暂停", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)

        self.restore_button = tk.Button(control_frame, text="恢复站位", command=self.restore_initial_positions, state=tk.DISABLED)
        self.restore_button.pack(side=tk.LEFT, padx=5)

        self.game_over_label = tk.Label(control_frame, text="", fg="red") # 模拟结果信息标签
        self.game_over_label.pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack(pady=10)
        self.draw_grid()  # 先画一次网格

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    # 暂停/继续逻辑
    def toggle_pause(self):
        if self.state_machine.state == AppState.SIMULATING:
            self.state_machine.transition_to(AppState.PAUSED)
        elif self.state_machine.state == AppState.PAUSED:
            self.state_machine.transition_to(AppState.SIMULATING)
            self.simulate()
        # if self.simulating:
        #     self.is_paused = not self.is_paused
        #     if self.is_paused:
        #         self.pause_button.config(text="继续")
        #         self.deploy_button.config(state=tk.NORMAL)  # 允许在暂停时部署
        #         self.restore_button.config(state=tk.NORMAL)
        #     else:
        #         self.pause_button.config(text="暂停")
        #         self.deploy_button.config(state=tk.DISABLED)
        #         self.simulate()  # 恢复模拟

    def apply_speed(self):
        try:
            new_speed = float(self.speed_entry.get())
            if new_speed <= 0:
                raise ValueError("速度必须大于0")
            self.speed_multiplier = new_speed
        except ValueError as e:
            messagebox.showerror("错误", f"无效的速度值: {e}")
            self.speed_entry.delete(0, tk.END)
            self.speed_entry.insert(0, str(self.speed_multiplier))  # REMOVED_TEAM_INTERFACE: Was 1.0, should be current multiplier

    def draw_grid(self):
        danger_zone = 0
        if self.battle_field and self.battle_field.danger_zone_size() > 0:
            danger_zone = min(self.battle_field.danger_zone_size(), self.grid_height / 2 + 1)
            self.canvas.create_rectangle(0, 0, self.canvas_width, danger_zone * self.cell_size, fill='#cccc00')
            self.canvas.create_rectangle(0, 0, danger_zone * self.cell_size, self.canvas_height, fill='#cccc00')
            self.canvas.create_rectangle(self.canvas_width, 0, self.canvas_width - danger_zone * self.cell_size, self.canvas_height, fill='#cccc00')
            self.canvas.create_rectangle(self.canvas_width, self.canvas_height, 0, self.canvas_height - danger_zone * self.cell_size, fill='#cccc00')

        for i in range(self.grid_width + 1):
            x = i * self.cell_size
            self.canvas.create_line(x, 0, x, self.canvas_height, fill='lightgray')  # 淡灰色网格线
        for i in range(self.grid_height + 1):
            y = i * self.cell_size
            self.canvas.create_line(0, y, self.canvas_width, y, fill='lightgray')

    def refresh_canvas_display(self):
        if not self.canvas: return
        self.canvas.delete("all")
        self.draw_grid()

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
                    messagebox.showerror(f"错误", f"怪物名 {monster.name} 在 REVERSE_MONSTER_MAPPING 中未找到!")
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

    def draw_unit(self, unit, monster : 'Monster'):  # monster_obj 用于获取更详细的状态，如 AttackState
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

        attack_state_color = "yellow"
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
            if not messagebox.askyesno("确认", "是否中断当前战斗？"):
                return
        # self.setup_phase_active = True
        # self.simulating = False
        

        # self.confirm_start_button.config(state=tk.NORMAL)
        # self.deploy_button.config(text="重新部署")
        self.game_over_label.config(text="") # 清空游戏结束信息标签
        print("进入部署阶段。使用 self.battle_data 初始化战场。")
        self.state_machine.transition_to(AppState.SETUP)
        self.init_battlefield_for_setup()


    def start_actual_simulation(self):
        # if not self.setup_phase_active:
        #     messagebox.showinfo("提示", "请先通过'部署怪物'加载初始设置。")
        #     return
        # if not self.battle_field.monsters:
        #     messagebox.showinfo("提示", "战场上没有怪物，无法开始模拟。")
        #     return
        if self.state_machine.state != AppState.SETUP:
            messagebox.showinfo("提示", "请先通过'部署怪物'加载初始设置。")
            return
        self.initial_battlefield = copy.deepcopy(self.battle_field)
        # self.setup_phase_active = False
        # self.simulating = True
        # self.is_paused = False  # 新增
        # self.pause_button.config(state=tk.NORMAL)  # 启用暂停按钮
        # self.confirm_start_button.config(state=tk.DISABLED)
        # self.restore_button.config(state=tk.NORMAL)  # 启用恢复按钮
        # self.deploy_button.config(text="部署怪物")
        # self.battle_field.monster_temporal_area_left.clear()  # Ensure these are cleared
        # self.battle_field.monster_temporal_area_right.clear()
        # self.battle_field.current_spawn_left = 0
        # self.battle_field.current_spawn_right = 0
        print("战斗模拟开始！")
        self.state_machine.transition_to(AppState.SIMULATING)
        self.game_over_label.config(text="") # 清空游戏结束信息标签
        self.simulate()

    def simulate(self):
        if self.state_machine.state != AppState.SIMULATING:  # 新增暂停检查
            return
        result = None
        for _ in range(int(self.speed_multiplier)):
            result = self.battle_field.run_one_frame()
            if result is not None:
                break
        self.refresh_canvas_display()
        if result is not None:
            # self.simulating = False
            # self.confirm_start_button.config(state=tk.NORMAL)
            # self.restore_button.config(state=tk.NORMAL)  # 启用恢复按钮
            # self.deploy_button.config(text="部署怪物")
            self.state_machine.transition_to(AppState.ENDED)
            winner_faction = result
            game_over_message = ""
            if winner_faction == Faction.LEFT:
                game_over_message = "左方胜利！"
                self.game_over_label.config(text=game_over_message, fg="red")
            elif winner_faction == Faction.RIGHT:
                game_over_message = "右方胜利！"
                self.game_over_label.config(text=game_over_message, fg="blue")
            else:
                game_over_message = f"游戏结束，结果: {result}"
                self.game_over_label.config(text=game_over_message, fg="black")
        else:
            interval = max(1, 33)
            self.simulation_id = self.master.after(interval, self.simulate)

    def show_result(self, message):
        # messagebox.showinfo("游戏结束", message) # 移除 messagebox 调用
        self.game_over_label.config(text=message) # 在标签中显示游戏结束信息

    def clear_sandbox(self):
        if self.state_machine.state == AppState.SIMULATING:
            self.master.after_cancel(self.simulation_id)
        
        self.state_machine.transition_to(AppState.INITIAL)
        self.units = []
        self.battle_field = Battlefield(self.monster_data)
        self.selected_monster_for_drag = None
        self.selected_team = None  # REMOVED_TEAM_INTERFACE: Still useful to reset these
        self.selected_unit_id = None  # REMOVED_TEAM_INTERFACE: Still useful to reset these
        # self.confirm_start_button.config(state=tk.DISABLED)
        # self.restore_button.config(state=tk.DISABLED)
        # self.deploy_button.config(text="部署怪物")
        # self.deploy_button.config(state=tk.NORMAL)
        # self.is_paused = False
        # self.pause_button.config(text="暂停", state=tk.DISABLED)
        if self.canvas:
            self.refresh_canvas_display()
        self.game_over_label.config(text="") # 清空游戏结束信息标签
        print("战场已清空。")

    def restore_initial_positions(self):
        if not self.initial_battlefield:
            messagebox.showinfo("提示", "没有可恢复的初始站位")
            return
        # 用深拷贝的初始状态替换当前战场
        self.battle_field = copy.deepcopy(self.initial_battlefield)
        # 保持时间连续性
        self.battle_field.gameTime = self.initial_battlefield.gameTime  
        # 强制进入部署状态
        if self.state_machine.state in [AppState.PAUSED, AppState.ENDED]:
            self.state_machine.transition_to(AppState.SETUP)
        # self.is_paused = True
        # if self.is_paused:
        #     self.pause_button.config(text="继续")
        #     self.deploy_button.config(state=tk.NORMAL)  # 允许在暂停时部署
        #     self.simulating = True
        #     self.pause_button.config(state=tk.NORMAL)  # 启用暂停按钮
        #     self.confirm_start_button.config(state=tk.DISABLED)
        #     self.restore_button.config(state=tk.NORMAL)  # 启用恢复按钮
        #     self.deploy_button.config(text="部署怪物")
        self.refresh_canvas_display()

def main():
    root = tk.Tk()
    root.withdraw()

    initial_battle_setup = {
        "left": {"大喷蛛": 2},
        "right": {"矿脉守卫": 3}
    }

    # 尝试从 stdin 读取 JSON
    try:
        # 检查 stdin 是否有数据（非交互式模式）
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

    if len(initial_battle_setup["left"]) == 0 or len(initial_battle_setup["left"]) == 0:
        messagebox.showerror("怪物为空！", "怪物为空！")
        return

    for key in initial_battle_setup["left"]:
        match key:
            case "矿脉守卫"|"鳄鱼"|"凋零萨卡兹"|"1750哥"|"高能源石虫":
                messagebox.showerror("警告：存在已知问题怪物！", key)
    for key in initial_battle_setup["right"]:
        match key:
            case "矿脉守卫"|"鳄鱼"|"凋零萨卡兹"|"1750哥"|"高能源石虫":
                messagebox.showerror("警告：存在已知问题怪物！", key)

    app = SandboxSimulator(root, initial_battle_setup)
    app.master.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()
