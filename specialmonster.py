class SpecialMonsterHandler:
    """
    可按照格式自行添加特殊怪物的留言信息：
    格式：
    怪物ID: {
        'name': '怪物名称',  # 可留空
        'win_message': '胜利时显示的消息',  # 可留空
        'lose_message': '失败时显示的消息',  # 可留空
    }
    """
    def __init__(self):
        self.special_monsters = {
            1: {
                'name': '狗神',
                'win_message': "全军出击，我咬死你！",
                'lose_message': "牙崩了牙崩了"
            },
            50: {
                'name': '王婷',
                'win_message': "敢玩王婷吗?敢玩王婷吗?敢玩王婷吗",
                'lose_message': "敢玩王婷吗?敢玩王婷吗?敢玩王婷吗"
            },
            16: {
                'name': 'vvan',
                'win_message': "相信完美团！",
                'lose_message': "别信vvan！"
            },
            18: {
                'name': '链神',
                'win_message': "相信链神！",
                'lose_message': ""
            },
            26: {
                'name': '鼠鼠',
                'win_message': "相信鼠鼠一次好不好QAQ",
                'lose_message': ""
            },
            53: {
                'name': '门',
                'win_message': "你相信门能创造奇迹吗？",
                'lose_message': "你相信门能创造奇迹吗？"
            },
            28: {
                'name': '杰斯顿·威廉姆斯',
                'win_message': "幽默杰斯顿",
                'lose_message': "幽默杰斯顿"
            },
            56: {
                'name': '酒桶',
                'win_message': "酒桶能全闪了吗?",
                'lose_message': "酒桶能全闪了吗?"
            }
        }

    def check_special_monsters(self, app, winner):
        messages = []
        
        for monster_id, config in self.special_monsters.items():
            left_has = app.left_monsters[str(monster_id)].text().isdigit() and int(app.left_monsters[str(monster_id)].text()) > 0
            right_has = app.right_monsters[str(monster_id)].text().isdigit() and int(app.right_monsters[str(monster_id)].text()) > 0
            
            if left_has or right_has:
                if winner == "左方" and left_has and config['win_message']:
                    messages.append(config['win_message'])
                elif winner == "右方" and right_has and config['win_message']:
                    messages.append(config['win_message'])
                elif winner == "右方" and left_has and config['lose_message']:
                    messages.append(config['lose_message'])
                elif winner == "左方" and right_has and config['lose_message']:
                    messages.append(config['lose_message'])
        
        return "\n".join(messages)