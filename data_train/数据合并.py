import csv
import shutil
import sys
from pathlib import Path
import unpackage

# 获取项目根目录
base_dir = Path(__file__).resolve().parent
project_root = base_dir.parent

# 将项目根目录添加到 sys.path 以便导入 config
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import MONSTER_COUNT, FIELD_FEATURE_COUNT

def get_expected_header():
    """根据 start_auto_fetch 的逻辑生成预期表头"""
    if FIELD_FEATURE_COUNT > 0:
        header = [f"{i+1}L" for i in range(MONSTER_COUNT)]
        header += [f"{i+1}LF" for i in range(MONSTER_COUNT, MONSTER_COUNT + FIELD_FEATURE_COUNT)]
        header += [f"{i+1}R" for i in range(MONSTER_COUNT)]
        header += [f"{i+1}RF" for i in range(MONSTER_COUNT, MONSTER_COUNT + FIELD_FEATURE_COUNT)]
        header += ["Result", "ImgPath"]
    else:
        header = [f"{i+1}L" for i in range(MONSTER_COUNT)]
        header += [f"{i+1}R" for i in range(MONSTER_COUNT)]
        header += ["Result", "ImgPath"]
    return header

def read_csv_data(filepath):
    """读取CSV文件，自动检测编码并返回表头和数据行"""
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'big5', 'latin1']
    for encoding in encodings:
        try:
            with open(filepath, 'r', newline='', encoding=encoding) as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    return None, [], encoding
                data = list(reader)
                return header, data, encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法以支持的编码读取文件 {filepath}")

def main():
    target_images_dir = base_dir / 'images'
    target_csv_path = base_dir / 'arknights.csv'

    # 清理旧数据
    if target_images_dir.exists():
        shutil.rmtree(target_images_dir)
    target_csv_path.unlink(missing_ok=True)

    target_images_dir.mkdir(parents=True, exist_ok=True)

    expected_header = get_expected_header()
    img_path_idx = expected_header.index("ImgPath")
    merged_data = []
    seen_img_paths = set()

    # 扫描子目录
    scan_dirs = [base_dir]
    tmp_dir = base_dir / 'tmp'
    if tmp_dir.exists() and tmp_dir.is_dir():
        scan_dirs.append(tmp_dir)

    for s_dir in scan_dirs:
        for sub_dir in s_dir.iterdir():
            if not sub_dir.is_dir() or sub_dir.name in ['images', 'tmp', 'package', '__pycache__']:
                continue
            
            csv_path = sub_dir / 'arknights.csv'
            if not csv_path.exists():
                continue

            try:
                current_header, data, encoding = read_csv_data(csv_path)
                if current_header is None:
                    continue

                if current_header != expected_header:
                    print(f"目录 {sub_dir.name}: arknights.csv 的表头不符合预期格式 (共 {len(data)} 行)")
                
                # 合并图片
                src_images_dir = sub_dir / 'images'
                if src_images_dir.exists() and src_images_dir.is_dir():
                    for src_img in src_images_dir.iterdir():
                        if src_img.is_file():
                            dst_img = target_images_dir / src_img.name
                            shutil.move(str(src_img), str(dst_img))
                
                # 增加重复检查
                added_count = 0
                skip_count = 0
                for row in data:
                    img_path = row[img_path_idx]
                    if img_path not in seen_img_paths:
                        merged_data.append(row)
                        seen_img_paths.add(img_path)
                        added_count += 1
                    else:
                        skip_count += 1
                
                print(f"成功合并目录: {sub_dir.name} (编码: {encoding}, 新增: {added_count}, 重复跳过: {skip_count})")
            except Exception as e:
                print(f"处理目录 {sub_dir.name} 时出错: {e}")

    if merged_data:
        with open(target_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(expected_header)
            writer.writerows(merged_data)
        print(f"\n合并完成，共 {len(merged_data)} 条记录 -> {target_csv_path}")
    else:
        print("\n未找到有效的 CSV 数据。")

if __name__ == '__main__':
    unpackage.unpackage()  # 解压数据包
    main()
