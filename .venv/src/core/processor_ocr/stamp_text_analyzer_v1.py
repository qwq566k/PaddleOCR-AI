# -*- coding: utf-8 -*-
"""
OCR签章/签名识别与标注程序（基于PaddleOCR输出）
功能：
1. 解析PaddleOCR输出JSON，提取签名、签章和关键词内容及其坐标；
2. 支持自定义关键词提取；
3. 自动生成图像标注结果并保存为可视化图像（使用OpenCV绘制）；
4. 从签名文本中提取签名人名称。
"""
import json
import cv2
import re
import numpy as np
from typing import List, Tuple, Dict
import os

# 签名和签章关键词定义
SIGNATURE_KEYWORDS = ["签名", "编制", "审核", "批准"]
STAMP_KEYWORDS = ["章", "印", "盖章"]

# 颜色配置
COLOR_SIGNATURE = (0, 255, 0)  # 绿色
COLOR_STAMP = (0, 0, 255)      # 红色
COLOR_KEYWORD = (255, 0, 0)    # 蓝色
COLOR_NAME = (0, 255, 255)     # 黄色，用于签名人名
FONT = cv2.FONT_HERSHEY_SIMPLEX

# 加载OCR JSON数据
def load_ocr_json(path: str) -> Dict:
    """加载 PaddleOCR 输出的 JSON 文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 解析关键词匹配项
def find_keywords(texts: List[str], boxes: List[List[int]], keywords: List[str]) -> List[Tuple[str, List[int]]]:
    """在文本列表中查找包含任一关键词的条目，返回文本及其包围盒"""
    results = []
    for text, box in zip(texts, boxes):
        for kw in keywords:
            if kw in text:
                results.append((text, box))
                break
    return results

# 从签名文本中提取人名
def extract_person_name(text: str) -> str:
    """提取签名文本中紧跟在关键词后面的姓名，支持常见分隔符：：: -"""
    # 匹配关键词后可能的分隔符及姓名
    pattern = r"(?:签名|编制|审核|批准)[:：\-\s]?([\u4e00-\u9fa5A-Za-z0-9]+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return ""

# 提取OCR信息
def extract_ocr_info(ocr_data: Dict, custom_keywords: List[str] = None) -> Dict:
    """提取签名/签章、关键词及签名人名信息"""
    texts = ocr_data.get("rec_texts", [])
    boxes = ocr_data.get("rec_boxes", [])

    # 基本匹配
    signatures = find_keywords(texts, boxes, SIGNATURE_KEYWORDS)
    stamps = find_keywords(texts, boxes, STAMP_KEYWORDS)
    custom_matches = find_keywords(texts, boxes, custom_keywords) if custom_keywords else []

    # 签名人名提取
    signature_names = []
    for text, box in signatures:
        name = extract_person_name(text)
        if name:
            signature_names.append((name, box))

    return {
        "signatures": signatures,
        "stamps": stamps,
        "custom_matches": custom_matches,
        "signature_names": signature_names
    }

# 图像标注：在图像上标注签名、签章、关键词及签名人名

def draw_annotations(image_path: str, ocr_info: Dict, output_path: str):
    """读取图像并绘制标注框和标签，保存结果"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像：{image_path}")

    def draw_boxes(info_list, color, label_prefix):
        for text, box in info_list:
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label_prefix}: {text}", (x1, y1 - 10), FONT, 0.5, color, 1, cv2.LINE_AA)

    # 绘制签名、签章、关键词
    draw_boxes(ocr_info["signatures"], COLOR_SIGNATURE, "签名")
    draw_boxes(ocr_info["stamps"], COLOR_STAMP, "签章")
    draw_boxes(ocr_info["custom_matches"], COLOR_KEYWORD, "关键词")

    # 绘制签名人名
    for name, box in ocr_info.get("signature_names", []):
        x1, y1, x2, y2 = box
        cv2.putText(image, f"姓名: {name}", (x1, y2 + 20), FONT, 0.5, COLOR_NAME, 1, cv2.LINE_AA)

    cv2.imwrite(output_path, image)
    print(f"已保存标注图像至: {output_path}")

# 打印结果信息
def print_results(results: Dict):
    """在控制台输出提取结果"""
    print("\n[签名相关内容及位置]:")
    for text, box in results['signatures']:
        print(f" - 内容: {text}, 位置: {box}")

    print("\n[签名人名提取]:")
    for name, box in results.get('signature_names', []):
        print(f" - 姓名: {name}, 关联签名位置: {box}")

    print("\n[签章相关内容及位置]:")
    for text, box in results['stamps']:
        print(f" - 内容: {text}, 位置: {box}")

    if results['custom_matches']:
        print("\n[自定义关键词匹配结果]:")
        for text, box in results['custom_matches']:
            print(f" - 内容: {text}, 位置: {box}")

# 示例调用
if __name__ == "__main__":
    json_path = "output/page_0004_res.json"
    image_path = "output/page_0004.png"
    output_path = "output/annotated_page_0004_with_names.png"

    # 加载OCR数据
    ocr_data = load_ocr_json(json_path)
    custom_keywords = ["培训", "溯源", "日期"]

    # 信息提取
    results = extract_ocr_info(ocr_data, custom_keywords)
    print_results(results)

    # 自动图像标注
    draw_annotations(image_path, results, output_path)
