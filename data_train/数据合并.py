import sys
import zipfile
import csv
import io
import shutil
import pandas as pd
from pathlib import Path

# 获取项目根目录
base_dir = Path(__file__).resolve().parent
project_root = base_dir.parent

# 将项目根目录添加到 sys.path 以便导入 config
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

#from config import MONSTER_COUNT, FIELD_FEATURE_COUNT config会读取图片导致非常慢

def load_monster_data():
    monster_data = pd.read_csv('monster_greenvine.csv', index_col="id", encoding='utf-8-sig')
    return monster_data

MONSTER_DATA = load_monster_data()

# 全局变量
MONSTER_COUNT = len(MONSTER_DATA)
FIELD_FEATURE_COUNT = 0


def get_expected_header():
    """根据配置生成预期表头"""
    if FIELD_FEATURE_COUNT > 0:
        header = [f"{i + 1}L" for i in range(MONSTER_COUNT)]
        header += [f"{i + 1}LF" for i in range(MONSTER_COUNT, MONSTER_COUNT + FIELD_FEATURE_COUNT)]
        header += [f"{i + 1}R" for i in range(MONSTER_COUNT)]
        header += [f"{i + 1}RF" for i in range(MONSTER_COUNT, MONSTER_COUNT + FIELD_FEATURE_COUNT)]
        header += ["Result", "ImgPath"]
    else:
        header = [f"{i + 1}L" for i in range(MONSTER_COUNT)]
        header += [f"{i + 1}R" for i in range(MONSTER_COUNT)]
        header += ["Result", "ImgPath"]
    return header


def read_csv_from_zip(zip_ref, csv_filename):
    """从 ZIP 文件流中直接读取 CSV 数据，尝试多种编码"""
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'big5', 'latin1']

    for encoding in encodings:
        try:
            with zip_ref.open(csv_filename) as f:
                text_f = io.TextIOWrapper(f, encoding=encoding, newline='')
                reader = csv.reader(text_f)
                try:
                    header = next(reader)
                except StopIteration:
                    return None, [], encoding
                data = list(reader)
                return header, data, encoding
        except (UnicodeDecodeError, io.UnsupportedOperation):
            continue

    raise ValueError(f"无法以支持的编码读取压缩包内的文件 {csv_filename}")


def process_archives(merge_images=True, extract_result_images=False):
    package_dir = base_dir / "package"
    target_images_dir = base_dir / 'images'
    target_csv_path = base_dir / 'arknights.csv'

    # 1. 目录准备 (移除旧的清空逻辑，保证增量更新)
    if not package_dir.exists():
        print(f"未找到压缩包目录: {package_dir}")
        return

    if merge_images:
        # 直接确保目录存在，不删除已有内容
        target_images_dir.mkdir(parents=True, exist_ok=True)

    expected_header = get_expected_header()
    img_path_idx = expected_header.index("ImgPath")

    # 2. 构建现有 CSV 数据的索引集合
    seen_csv_img_paths = set()
    is_csv_initialized = False # 用于判断是否需要写入表头

    if target_csv_path.exists():
        print(f"检测到已存在的 {target_csv_path.name}，正在读取历史记录构建索引...")
        try:
            with open(target_csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                    if header == expected_header:
                        is_csv_initialized = True
                        for row in reader:
                            if len(row) > img_path_idx:
                                seen_csv_img_paths.add(row[img_path_idx])
                        print(f"-> 成功加载 {len(seen_csv_img_paths)} 条历史记录的索引。")
                    else:
                        print("-> 警告：已存在 CSV 的表头与配置不符，将创建新文件或覆写。")
                        target_csv_path.unlink() # 表头不对则删除重建
                except StopIteration:
                    # 文件为空
                    pass
        except Exception as e:
            print(f"读取历史 CSV 失败，将重新生成: {e}")
            target_csv_path.unlink(missing_ok=True)

    # 定义可能的文件后缀，涵盖常见大小写
    possible_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']

    zip_files = list(package_dir.glob("*.zip"))
    print(f"\n找到 {len(zip_files)} 个压缩包，准备进行增量处理...")

    total_added_rows = 0
    total_extracted_imgs = 0

    # 3. 遍历压缩包
    for zip_path in zip_files:
        print(f"\n正在处理压缩包: {zip_path.name}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # 建立 namelist 的 O(1) 查找集合
                zip_namelist_set = set(zf.namelist())
                
                # ==========================================
                # 任务 1：完全独立处理 CSV 文件
                # ==========================================
                csv_members = [m for m in zip_namelist_set if m.endswith('arknights.csv')]

                zip_added_csv_count = 0
                
                for csv_member in csv_members:
                    header, data, encoding = read_csv_from_zip(zf, csv_member)

                    if header is None or header != expected_header:
                        print(f"  [跳过] {csv_member}: 表头不符合预期格式")
                        continue

                    zip_new_csv_rows = []
                    skip_csv_count = 0

                    for row in data:
                        img_path = row[img_path_idx]

                        if img_path not in seen_csv_img_paths:
                            seen_csv_img_paths.add(img_path)
                            zip_new_csv_rows.append(row)
                        else:
                            skip_csv_count += 1

                    # 追加写入 CSV
                    if zip_new_csv_rows:
                        mode = 'a' if is_csv_initialized else 'w'
                        with open(target_csv_path, mode, newline='', encoding='utf-8-sig') as f:
                            writer = csv.writer(f)
                            if not is_csv_initialized:
                                writer.writerow(expected_header)
                                is_csv_initialized = True
                            writer.writerows(zip_new_csv_rows)
                    
                    zip_added_csv_count += len(zip_new_csv_rows)
                    print(f"  -> {csv_member} (编码: {encoding})")
                    print(f"     CSV: 新增 {len(zip_new_csv_rows)} 条，重复跳过 {skip_csv_count} 条")

                total_added_rows += zip_added_csv_count

                # ==========================================
                # 任务 2：完全独立处理图片文件
                # ==========================================
                zip_extracted_img_count = 0
                zip_skip_img_count = 0

                if merge_images:
                    for member in zip_namelist_set:
                        # 过滤出所有图片文件
                        if member.lower().endswith(tuple(possible_extensions)):
                            # 排除 Mac 系统压缩可能产生的隐藏文件干扰
                            if "__MACOSX" in member:
                                continue
                                
                            filename = Path(member).name
                            is_result_img = filename.rsplit('.', 1)[0].endswith('_result')

                            # 根据参数决定是否跳过 result 图
                            if is_result_img and not extract_result_images:
                                continue

                            target_img_path = target_images_dir / filename
                            
                            # 增量判断：如果本地没有，则解压提取
                            if target_img_path.exists():
                                zip_skip_img_count += 1
                            else:
                                try:
                                    with zf.open(member) as source_file:
                                        with open(target_img_path, 'wb') as target_file:
                                            shutil.copyfileobj(source_file, target_file)
                                    zip_extracted_img_count += 1
                                except Exception as e:
                                    print(f"  [错误] 提取图片 {member} 失败: {e}")
                    
                    print(f"     IMG: 提取新图 {zip_extracted_img_count} 张，已存在跳过 {zip_skip_img_count} 张")
                    total_extracted_imgs += zip_extracted_img_count

        except zipfile.BadZipFile:
            print(f"压缩包损坏，跳过: {zip_path.name}")
        except Exception as e:
            print(f"处理压缩包 {zip_path.name} 时出错: {e}")

    print(f"\n全部处理完成！")
    print(f"总计新增 CSV 记录: {total_added_rows} 条")
    if merge_images:
        print(f"总计提取新图片: {total_extracted_imgs} 张")


if __name__ == '__main__':
    merge_imgs = False # 设置为 True 则提取阵容图
    extract_res_imgs = False  # 设置为 True 则同时提取带有 _result 的结果图
    process_archives(merge_images=merge_imgs, extract_result_images=extract_res_imgs)
