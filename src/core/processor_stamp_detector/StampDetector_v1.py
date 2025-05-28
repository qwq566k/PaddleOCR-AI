import os
import cv2
import numpy as np
import logging
from typing import List, Dict
from PIL import Image
from paddleocr import PaddleOCR
from config.logger_config import get_logger
from config.ocr_settings import OcrConfig


class StampDetector:
    def __init__(self):
        self.logger = get_logger('StampDetector')
        self.config = OcrConfig()
        self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='ch')  # 启用角度分类
        self.min_stamp_area = 500  # 可配置参数移至此处
        self.text_confidence = 0.6  # OCR置信度阈值
        self.red_lower_ranges = [  # 红色HSV范围可配置化
            (0, 100, 100), (160, 100, 100)
        ]
        self.red_upper_ranges = [
            (10, 255, 255), (180, 255, 255)
        ]


    def _detect_stamps(self, image_path: str, config) -> List[Dict]:
        """主检测流程：检测红色区域→形状验证→文字关联验证→保存结果"""
        try:
            # 加载图像
            img = Image.open(image_path).convert('RGB')
            # 转为NumPy数组
            np_img = np.array(img)
            # 最终签章结果
            stamps = []

            # 1. 检测红色区域
            red_regions = self._detect_red_regions(np_img)
            if not red_regions:
                self.logger.info("未检测到红色区域")
                return []

            # 2. 形状验证（过滤非印章轮廓）
            valid_stamps = self._validate_shape(red_regions)
            if not valid_stamps:
                self.logger.info("红色区域未通过形状验证")
                return []

            # 3. 文字关联验证（检查周边文本）
            final_stamps = self._verify_text_proximity(valid_stamps, np_img)
            if not final_stamps:
                self.logger.info("未通过文字关联验证")
                return []

            # 4. 保存签章图像
            for idx, stamp in enumerate(final_stamps):
                # 裁剪签章区域
                stamp_img = img.crop(stamp['bbox'])
                save_dir = os.path.join(self.config.output_dir, "stamps")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{os.path.basename(image_path)}_{idx}.png")
                stamp_img.save(save_path)
                stamps.append({"position": stamp['bbox'], "image_path": save_path})

            return stamps

        except Exception as e:
            self.logger.error(f"签章检测失败: {image_path} - {str(e)}")
            return []

    def _detect_red_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """通过HSV颜色空间检测红色区域，返回轮廓列表.添加形态学操作减少噪声"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        masks = []
        # 合并两个红色区域的掩膜
        for lower, upper in zip(self.red_lower_ranges, self.red_upper_ranges):
            masks.append(cv2.inRange(hsv, np.array(lower), np.array(upper)))
        combined_mask = cv2.bitwise_or(masks[0], masks[1])

        # 形态学操作：先膨胀后腐蚀，填充小孔洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _validate_shape(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """通过面积和多边形近似验证轮廓是否为签章形状，优化：添加圆形度验证"""
        valid = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_stamp_area:
                continue

            # 计算圆形度：4π*面积/周长²（圆为1，其他形状接近0）
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            if 0.7 < circularity < 1.2:  # 允许轻微形变
                valid.append(cnt)
        return valid

    def _fast_ocr(self, image: np.ndarray):
        """优化：改进图像缩放策略"""
        try:
            h, w = image.shape[:2]
            scale = 1.0
            if max(h, w) > 1024:
                scale = 1024 / max(h, w)
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            return self._parse_ocr_results(self.ocr_engine.ocr(image))
        except Exception as e:
            self.logger.error(f"OCR失败: {str(e)}", exc_info=True)
            return []

    def _parse_ocr_results(self, results):
        """提取OCR结果公共方法"""
        texts = []
        if not results:
            return texts
        for line in results:
            if not line or len(line) < 2:
                continue
            text_info = line[1]
            if text_info[1] >= self.text_confidence:
                texts.append((text_info[0], text_info[1], line[0]))
        return texts

    def _verify_text_proximity(self, stamps: List[np.ndarray], image: np.ndarray) -> List[Dict]:
        """修复：处理轮廓转换边界框错误"""
        verified = []
        keywords = ["公司", "汽车", "集团", "公章"]
        for cnt in stamps:
            x, y, w, h = cv2.boundingRect(cnt)  # 直接使用轮廓生成边界框
            roi = self._get_safe_roi(image, x, y, w, h, scale=1.5)

            ocr_results = self._fast_ocr(roi)
            if any(keyword in text for text, _, _ in ocr_results for keyword in keywords):
                verified.append({
                    "bbox": (x, y, x + w, y + h),
                    "related_texts": ocr_results
                })
        return verified

    def _get_safe_roi(self, image: np.ndarray, x: int, y: int, w: int, h: int, scale: float):
        """安全获取扩展区域，防止越界"""
        roi_x1 = max(0, int(x - w * scale))
        roi_y1 = max(0, int(y - h * scale))
        roi_x2 = min(image.shape[1], int(x + w * (1 + scale)))
        roi_y2 = min(image.shape[0], int(y + h * (1 + scale)))
        return image[roi_y1:roi_y2, roi_x1:roi_x2]


if __name__ == '__main__':
    detector = StampDetector()
    results = detector._detect_stamps('C:/log/page_0001.png', OcrConfig())
    print(results)