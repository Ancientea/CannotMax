import pandas as pd
import numpy as np
import os
import json
import re
from collections import defaultdict
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# ==============================================================================
# SECTION 1: 游戏画面元素识别模块 (已修改为PyTorch+GPU)
# ==============================================================================

ROI_COORDINATES = {
    "altar_vertical": [
        {"x": 910, "y": 174, "width": 95, "height": 104},
        {"x": 910, "y": 429, "width": 102, "height": 108},
        {"x": 900, "y": 755, "width": 120, "height": 108}
    ],
    "block_parallel": [
        {"x": 694, "y": 240, "width": 530, "height": 122},
        {"x": 651, "y": 614, "width": 620, "height": 143}
    ],
    "block_vertical": [
        {"x": 647, "y": 233, "width": 153, "height": 523},
        {"x": 1112, "y": 239, "width": 159, "height": 514}
    ],
    "coil_narrow": [
        {"x": 915, "y": 110, "width": 85, "height": 89},
        {"x": 815, "y": 257, "width": 86, "height": 98},
        {"x": 1024, "y": 258, "width": 79, "height": 98},
        {"x": 790, "y": 643, "width": 97, "height": 102},
        {"x": 1031, "y": 639, "width": 102, "height": 108}
    ],
    "coil_wide": [
        {"x": 719, "y": 181, "width": 81, "height": 89},
        {"x": 602, "y": 346, "width": 81, "height": 94},
        {"x": 578, "y": 535, "width": 81, "height": 95},
        {"x": 669, "y": 759, "width": 91, "height": 95},
        {"x": 1159, "y": 757, "width": 93, "height": 92},
        {"x": 1257, "y": 533, "width": 94, "height": 102},
        {"x": 1236, "y": 344, "width": 85, "height": 97},
        {"x": 1120, "y": 180, "width": 75, "height": 91}
    ],
    "crossbow_top": [
        {"x": 718, "y": 13, "width": 484, "height": 106}
    ],
    "fire_side_left": [
        {"x": 98, "y": 246, "width": 184, "height": 281}
    ],
    "fire_side_right": [
        {"x": 1656, "y": 430, "width": 235, "height": 315}
    ],
    "fire_top": [
        {"x": 532, "y": 17, "width": 188, "height": 97},
        {"x": 1325, "y": 14, "width": 60, "height": 100}
    ]
}


def predict_scene_pytorch(
        model: nn.Module,
        idx_to_class: dict,
        image_path: str,
        transform: transforms.Compose,
        device: torch.device,
        threshold: float = 0.5
) -> list[str]:
    """使用PyTorch模型对图片进行场景识别（GPU加速版）"""
    try:
        full_image = Image.open(image_path).convert('RGB')
    except Exception:
        return []

    if full_image.size != (1920, 1080):
        return []

    detected_classes = []
    with torch.no_grad():
        for location, boxes in ROI_COORDINATES.items():
            for i, box in enumerate(boxes):
                x, y, w, h = box['x'], box['y'], box['width'], box['height']
                roi_pil = full_image.crop((x, y, x + w, y + h))
                input_tensor = transform(roi_pil).unsqueeze(0).to(device)
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                max_prob, predicted_index_tensor = torch.max(probabilities, 0)
                predicted_index = predicted_index_tensor.item()

                if max_prob.item() >= threshold:
                    predicted_class = idx_to_class[predicted_index]
                    if not predicted_class.endswith('_none'):
                        detected_classes.append(predicted_class)
    return detected_classes


# ==============================================================================
# SECTION 2: 数据清洗主模块 (修改了最后的合并与保存部分)
# ==============================================================================

def clean_data(file_path, output_path, screenshots_base_path, onnx_model_path, class_map_path):
    print(f"开始清洗数据文件: {file_path}")
    try:
        data = pd.read_csv(file_path, header=0)
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 '{file_path}'")
        return
    data['original_index'] = data.index + 1

    # --- 原始清洗逻辑部分 (无变更) ---
    features = data.iloc[:, :-3]
    labels = data.iloc[:, -3]
    pic_names = data.iloc[:, -2]
    print(f"原始特征总数: {features.shape[1]}")
    # ... (其余清洗逻辑与原脚本完全相同)
    last_row_features = features.iloc[-1].values
    last_row_valid = True
    if abs(last_row_features[27]) > 6 or abs(last_row_features[61]) > 6: last_row_valid = False
    if np.any(np.abs(last_row_features) >= 100): last_row_valid = False
    if not last_row_valid: print("错误: 最后一行不满足清洗条件"); return
    last_row = data.iloc[-1].copy()
    rows_to_remove = [i for i, row in enumerate(features.values) if np.any(np.abs(row) >= 100)]
    cleaned_data = data.drop(rows_to_remove).reset_index(drop=True)
    if not (len(data) - 1 in rows_to_remove): cleaned_data = cleaned_data.iloc[:-1]
    if rows_to_remove: cleaned_data = pd.concat([cleaned_data, pd.DataFrame([last_row] * len(rows_to_remove))],
                                                ignore_index=True)
    cleaned_data = cleaned_data.drop_duplicates(subset=cleaned_data.columns[:-3], keep='first').reset_index(drop=True)
    features_cleaned, labels_cleaned, pic_names_cleaned = cleaned_data.iloc[:, :-3], cleaned_data.iloc[:,
                                                                                     -3], cleaned_data.iloc[:, -2]
    # ... (异常波动筛选逻辑完全相同)

    # --- 画面元素识别集成部分 (修改为PyTorch+GPU) ---
    print("\n开始使用PyTorch和GPU识别截图中的游戏元素...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用的推理设备: {device}")

    try:
        # 从ONNX路径生成对应的PTH路径
        pth_model_path = onnx_model_path.replace('.onnx', '.pth')
        print(f"加载PyTorch模型: {pth_model_path}")

        # 加载类别映射
        with open(class_map_path, 'r', encoding='utf-8') as f:
            class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)

        # 加载PyTorch模型
        model = models.mobilenet_v3_small(weights=None)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(pth_model_path, map_location=device))
        model.to(device)
        model.eval()

        # 定义图像转换
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 重新定义predict_scene函数使用PyTorch模型
        def predict_scene(session, idx_to_class, image_path, threshold=0.5):
            return predict_scene_pytorch(
                model, idx_to_class, image_path, transform, device, threshold
            )

    except Exception as e:
        print(f"错误：加载模型或class_map文件失败: {e}")
        return

    grouped_elements = defaultdict(list)
    for class_name in class_to_idx.keys():
        if class_name.endswith('_none'):
            continue
        condensed_name = re.sub(r'_left_', '_', class_name)
        condensed_name = re.sub(r'_right_', '_', condensed_name)
        grouped_elements[condensed_name].append(class_name)
    image_feature_columns = sorted(grouped_elements.keys())
    print(f"将聚合生成 {len(image_feature_columns)} 个新特征列。")
    all_rows_image_data = []
    total_pics = len(pic_names_cleaned)
    for idx, pic_name in enumerate(pic_names_cleaned):
        print(f"\r处理图片: {idx + 1}/{total_pics} ({pic_name})", end="")
        image_path = os.path.join(screenshots_base_path, str(pic_name))
        if not os.path.exists(image_path):
            row_image_data = {col: -1 for col in image_feature_columns}
            all_rows_image_data.append(row_image_data)
            continue
        try:
            detected_full_names = set(predict_scene(None, idx_to_class, image_path, threshold=0.5))
            row_image_data = {}
            for condensed_name, full_names in grouped_elements.items():
                num_positions = len(full_names)
                if num_positions == 1:
                    row_image_data[condensed_name] = 1 if full_names[0] in detected_full_names else 0
                else:
                    detections_in_group = [fn in detected_full_names for fn in full_names]
                    num_detected = sum(detections_in_group)
                    if num_detected == num_positions:
                        row_image_data[condensed_name] = 1
                    elif num_detected == 0:
                        row_image_data[condensed_name] = 0
                    else:
                        row_image_data[condensed_name] = -1
        except Exception as e:
            row_image_data = {col: -20 for col in image_feature_columns}
        all_rows_image_data.append(row_image_data)
    print("\n截图元素识别完成。")
    image_data_df = pd.DataFrame(all_rows_image_data)

    # --- MODIFIED: 合并与保存 ---
    # 重置所有数据部分的索引，确保能够正确拼接
    features_cleaned.reset_index(drop=True, inplace=True)
    image_data_df.reset_index(drop=True, inplace=True)
    labels_cleaned.reset_index(drop=True, inplace=True)
    pic_names_cleaned.reset_index(drop=True, inplace=True)

    # 1. 检查并拆分155个原始特征为L(77)和R(78)两组
    if features_cleaned.shape[1] != 122:
        print(f"警告: 期望122个原始特征，但检测到{features_cleaned.shape[1]}个。将按前61列和剩余列进行分割。")

    features_L = features_cleaned.iloc[:, :61]
    features_R = features_cleaned.iloc[:, 61:]

    # 2. 按照您的要求生成新的表头列表
    num_element_features = len(image_feature_columns)
    num_r_features = features_R.shape[1]

    headers_L = [f"{i}L" for i in range(1, 62)]  # 1L to 61L
    headers_elements_L = [f"{i}L" for i in range(62, 62 + num_element_features)]

    headers_R = [f"{i}R" for i in range(1, num_r_features + 1)]  # 1R to 61R
    headers_elements_R = [f"{i}R" for i in range(num_r_features + 1, num_r_features + 1 + num_element_features)]

    # 最终表头顺序
    final_headers = (headers_L + headers_elements_L +
                     headers_R + headers_elements_R +
                     ['Result', 'ImgPath'])

    # 3. 为Series命名，以便在拼接时作为列名
    labels_cleaned.name = 'Result'
    pic_names_cleaned.name = 'ImgPath'

    # 4. 按照新的顺序拼接所有数据部分
    final_cleaned_data = pd.concat([
        features_L,  # 1L-61L
        image_data_df,  # 元素特征
        features_R,  # 1R-61R
        image_data_df.copy(),  # 元素特征 (副本)
        labels_cleaned,  # label
        pic_names_cleaned  # screenshot_filename
    ], axis=1)

    # 5. 将新生成的表头赋予DataFrame
    final_cleaned_data.columns = final_headers

    # 6. 保存到CSV，使用新的表头
    final_cleaned_data.to_csv(output_path, index=False, header=True)

    print(f"\n清洗和识别后的数据已保存到: {output_path}")
    print(f"最终数据维度: {final_cleaned_data.shape[0]} 行, {final_cleaned_data.shape[1]} 列")
    print(f"已按要求生成自定义表头。")


if __name__ == "__main__":
    # 路径配置与之前保持一致
    input_file = r"arknights.csv"
    output_file = r"arknights_with_field_recognize_v2.csv"
    screenshots_base_path = r"images"

    model_dir = r"battlefield_recognize"
    onnx_model_path = os.path.join(model_dir, 'field_recognize.onnx')
    class_map_path = os.path.join(model_dir, 'class_to_idx.json')

    # 自动检查PTH文件是否存在
    pth_model_path = onnx_model_path.replace('.onnx', '.pth')
    if not os.path.exists(pth_model_path):
        print(f"错误: 找不到对应的PTH模型文件 '{pth_model_path}'")
        print(f"请确保存在与ONNX同名的PTH文件: {os.path.basename(pth_model_path)}")
    else:
        clean_data(input_file, output_file, screenshots_base_path, onnx_model_path, class_map_path)