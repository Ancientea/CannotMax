with open(r'G:\314\CannotMax-main\main.py',encoding='utf-8') as f:
    content = f.read()

old = '''        # 恢复原始预测HTML
        if hasattr(self, '_batch_base_html') and self._batch_base_html:
            self.result_label.setText(self._batch_base_html)
        self.batch_sim_button.setText("批量模拟")'''
new = '''        # 停止模拟，保留最后胜率不清除
        self.batch_sim_button.setText("批量模拟")'''

if old in content:
    content = content.replace(old, new)
    with open(r'G:\314\CannotMax-main\main.py','w',encoding='utf-8') as f:
        f.write(content)
    print("OK - stop keeps winrate")
else:
    # show context
    idx = content.find('_stop_batch_sim')
    print(content[idx:idx+400])
