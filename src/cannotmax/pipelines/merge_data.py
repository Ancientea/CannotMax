import sys
import zipfile
import csv
import io
import shutil
import pandas as pd
from pathlib import Path

# 路径配置
base_dir = Path(__file__).resolve().parent
project_root = base_dir.parent.parent.parent.parent  # 项目根目录

# 数据目录结构
DATA_DIR = project_root / "data"
COMPRESSED_DIR = DATA_DIR / "compressed"
TARGET_CSV_PATH = DATA_DIR / "arknights.csv"
TARGET_IMAGES_DIR = DATA_DIR / "images"


def load_monster_data():
    monster_data = pd.read_csv('monster_greenvine.csv', index_col="id", encoding='utf-8-sig')
    return monster_data

MONSTER_DATA = load_monster_data()
MONSTER_COUNT = len(MONSTER_DATA)
FIELD_FEATURE_COUNT = 0


def get_expected_header():
    """生成预期表头"""
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
    """从 ZIP 读取 CSV"""
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030']
    for encoding in encodings:
        try:
            with zip_ref.open(csv_filename) as f:
                text_f = io.TextIOWrapper(f, encoding=encoding, newline='')
                reader = csv.reader(text_f)
                header = next(reader)
                data = list(reader)
                return header, data, encoding
        except (UnicodeDecodeError, io.UnsupportedOperation):
            continue
    raise ValueError(f"无法读取 {csv_filename}")


def process_archives(merge_images=True, extract_result_images=False):
    """处理 compressed 下的压缩包和 data 下的日期目录"""
    
    # 1. 目录准备
    COMPRESSED_DIR.mkdir(parents=True, exist_ok=True)
    if merge_images:
        TARGET_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    expected_header = get_expected_header()
    img_path_idx = expected_header.index("ImgPath")

    # 2. 加载现有数据索引
    seen_csv_img_paths = set()
    new_csv_rows = []

    if TARGET_CSV_PATH.exists():
        print(f"读取现有 {TARGET_CSV_PATH.name}...")
        try:
            with open(TARGET_CSV_PATH, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header == expected_header:
                    for row in reader:
                        if len(row) > img_path_idx:
                            seen_csv_img_paths.add(row[img_path_idx])
                    print(f"-> 已加载 {len(seen_csv_img_paths)} 条历史记录。")
        except Exception as e:
            print(f"读取失败：{e}")

    possible_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    total_csv_added = 0
    total_imgs_extracted = 0
    total_sources = 0

    # 3.1 处理 compressed/*.zip
    zip_files = list(COMPRESSED_DIR.glob("*.zip"))
    if zip_files:
        print(f"\n[阶段 1] 处理 {len(zip_files)} 个压缩包...")
        for zip_path in zip_files:
            print(f"\n压缩包：{zip_path.name}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    # 获取顶层日期目录
                    top_dirs = sorted(set(
                        Path(n).parts[0] for n in zf.namelist() 
                        if Path(n).parts and Path(n).parts[0] != "__MACOSX"
                    ))
                    
                    for dir_name in top_dirs:
                        print(f"  目录：{dir_name}")
                        total_sources += 1
                        
                        # 处理 CSV
                        csv_in_zip = f"{dir_name}/arknights.csv"
                        if csv_in_zip in zf.namelist():
                            try:
                                with zf.open(csv_in_zip) as f:
                                    text = io.TextIOWrapper(f, encoding='utf-8-sig', newline='')
                                    reader = csv.reader(text)
                                    header = next(reader, None)
                                    if header == expected_header:
                                        added = 0
                                        for row in reader:
                                            if len(row) > img_path_idx:
                                                img_path = row[img_path_idx]
                                                if img_path not in seen_csv_img_paths:
                                                    seen_csv_img_paths.add(img_path)
                                                    new_csv_rows.append(row)
                                                    added += 1
                                        print(f"    CSV: +{added}")
                                        total_csv_added += added
                            except Exception as e:
                                print(f"    CSV 错误：{e}")
                        
                        # 处理图片
                        if merge_images:
                            ext_count = 0
                            skip_count = 0
                            for member in zf.namelist():
                                if member.startswith(f"{dir_name}/") and member.lower().endswith(tuple(possible_extensions)):
                                    filename = Path(member).name
                                    is_result = filename.rsplit('.', 1)[0].endswith('_result')
                                    if is_result and not extract_result_images:
                                        continue
                                    target = TARGET_IMAGES_DIR / filename
                                    if not target.exists():
                                        try:
                                            with zf.open(member) as src:
                                                with open(target, 'wb') as dst:
                                                    shutil.copyfileobj(src, dst)
                                            ext_count += 1
                                        except Exception as e:
                                            print(f"    图片错误 {filename}: {e}")
                                    else:
                                        skip_count += 1
                            total_imgs_extracted += ext_count
                            print(f"    IMG: +{ext_count}, skip {skip_count}")
            except Exception as e:
                print(f"  压缩包错误：{e}")

    # 3.2 处理 data/下的日期目录
    date_dirs = [
        d for d in DATA_DIR.iterdir() 
        if d.is_dir() 
        and d.name not in ('compressed', 'images', '.git')
        and len(d.name) == 19  # 2026_04_26__18_47_37
        and d.name[4] == '_' and d.name[7] == '_' and d.name[10] == '_'
    ]
    
    if date_dirs:
        print(f"\n[阶段 2] 处理 {len(date_dirs)} 个日期目录...")
        for date_dir in sorted(date_dirs):
            print(f"\n目录：{date_dir.name}")
            total_sources += 1
            
            # CSV
            csv_path = date_dir / "arknights.csv"
            if csv_path.exists():
                try:
                    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                        reader = csv.reader(f)
                        header = next(reader, None)
                        if header == expected_header:
                            added = 0
                            for row in reader:
                                if len(row) > img_path_idx:
                                    img_path = row[img_path_idx]
                                    if img_path not in seen_csv_img_paths:
                                        seen_csv_img_paths.add(img_path)
                                        new_csv_rows.append(row)
                                        added += 1
                            print(f"  CSV: +{added}")
                            total_csv_added += added
                except Exception as e:
                    print(f"  CSV 错误：{e}")
            
            # 图片
            if merge_images:
                ext_count = 0
                skip_count = 0
                for img_path in date_dir.glob("*"):
                    if img_path.is_file() and img_path.suffix.lower() in possible_extensions:
                        filename = img_path.name
                        is_result = filename.rsplit('.', 1)[0].endswith('_result')
                        if is_result and not extract_result_images:
                            continue
                        target = TARGET_IMAGES_DIR / filename
                        if not target.exists():
                            try:
                                shutil.copy2(img_path, target)
                                ext_count += 1
                            except:
                                pass
                        else:
                            skip_count += 1
                total_imgs_extracted += ext_count
                if ext_count > 0:
                    print(f"  IMG: +{ext_count}, skip {skip_count}")

    # 4. 写入 CSV
    if new_csv_rows:
        mode = 'a' if TARGET_CSV_PATH.exists() and seen_csv_img_paths else 'w'
        with open(TARGET_CSV_PATH, mode, newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if mode == 'w':
                writer.writerow(expected_header)
            writer.writerows(new_csv_rows)
        print(f"\n已追加 {len(new_csv_rows)} 条到 {TARGET_CSV_PATH.name}")

    print(f"\n完成！处理源：{total_sources}, CSV: +{total_csv_added}, IMG: +{total_imgs_extracted}")


if __name__ == '__main__':
    merge_imgs = False
    extract_res_imgs = False
    process_archives(merge_images=merge_imgs, extract_result_images=extract_res_imgs)
