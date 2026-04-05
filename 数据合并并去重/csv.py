import csv

def read_csv_data(filename):
    """读取CSV文件，自动检测编码并验证表头是否为纯数字，返回表头、数据行（去重后的集合）和成功的编码"""
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'big5', 'latin1']  # 尝试的编码顺序
    for encoding in encodings:
        try:
            with open(filename, 'r', newline='', encoding=encoding) as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    raise ValueError(f"文件 {filename} 为空或无法读取内容")
                if all(field.strip().isdigit() for field in header):
                    data = {tuple(row) for row in reader}
                    return header, data, encoding
                else:
                    continue  # 表头不符合要求，尝试下一编码
        except UnicodeDecodeError:
            continue  # 解码失败，尝试下一编码
    raise ValueError(f"无法以支持表头为纯数字的编码读取文件 {filename}")

# 读取两个文件并处理可能的错误
try:
    header_a, data_a, encoding_a = read_csv_data('a.csv')
    header_b, data_b, encoding_b = read_csv_data('b.csv')
except ValueError as e:
    print(f"错误: {e}")
    exit()

# 验证表头一致性
if header_a != header_b:
    raise ValueError("错误：两个文件的表头不一致，无法合并")

# 合并数据并去重
merged_data = data_a | data_b

# 写入新文件，统一使用UTF-8编码
with open('arknights.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header_a)
    for row in merged_data:
        writer.writerow(row)

print(f"文件合并完成，结果已保存到 arknights.csv")
print(f"文件 a.csv 使用的编码是: {encoding_a}")
print(f"文件 b.csv 使用的编码是: {encoding_b}")
