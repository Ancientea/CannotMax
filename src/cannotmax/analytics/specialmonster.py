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
        }

    def check_special_monsters(self, left_monsters, right_monsters, winner):
        messages = []
        
        for monster_id, config in self.special_monsters.items():
            left_has = left_monsters[str(monster_id)].text().isdigit() and int(left_monsters[str(monster_id)].text()) > 0
            right_has = right_monsters[str(monster_id)].text().isdigit() and int(right_monsters[str(monster_id)].text()) > 0
            
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