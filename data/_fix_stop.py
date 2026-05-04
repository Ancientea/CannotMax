with open(r'G:\314\CannotMax-main\main.py', encoding='utf-8') as f:
    content = f.read()

# Fix _stop_batch_sim: don't restore HTML
old = '''        # 恢复原始预测HTML
        if hasattr(self, '_batch_base_html') and self._batch_base_html:
            self.result_label.setText(self._batch_base_html)
        self.batch_sim_button.setText("批量模拟")'''
new = '''        self.batch_sim_button.setText("批量模拟")'''
content = content.replace(old, new)

# Fix _start_batch_sim: also save HTML only if there's a prediction
old2 = '''        self._batch_base_html = self.result_label.text()  # 保存原始预测HTML'''
new2 = '''        self._batch_base_html = self.result_label.text()  # 保存原始预测HTML'''

with open(r'G:\314\CannotMax-main\main.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done - stop won't clear batch display")
