import os
import cv2
import numpy as np
from typing import List, Dict, Tuple
from math import pi
from PIL import Image
from paddleocr import PaddleOCR

from config.logger_config import get_logger
from config.ocr_settings import OcrConfig

class StampDetector:
    def __init__(self):
        self.logger = get_logger('StampDetector')
        self.config = OcrConfig()
        # 初始化PaddleOCR引擎（关闭角度分类，中文识别）
        self.ocr_engine = PaddleOCR(use_angle_cls=False, lang='ch')

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
            print(e)
            self.logger.error(f"签章检测失败: {image_path} - {str(e)}")
            return []

    def _detect_red_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """通过HSV颜色空间检测红色区域，返回轮廓列表"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # 定义红色范围（覆盖0-10和160-180两个区间）
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        # 合并两个红色区域的掩膜
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        # 提取轮廓（仅外部轮廓）
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _validate_shape(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """通过面积和多边形近似验证轮廓是否为签章形状"""
        valid = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            # 多边形近似
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

            # 过滤条件：面积>500且顶点数>=5（近似圆形）
            if area > 500 and len(approx) >= 5:
                valid.append(cnt)
        return valid

    def verify_stamps_and_text(self, final_stamps, img, image_path):
        """
        Verifies existence of stamps and extracts text inside each detected stamp region.

        Parameters:
            final_stamps (List[Dict]): List of detected stamp information, each with a 'bbox' (box or polygon).
            img (PIL.Image): Full image from which stamps were detected.
            image_path (str): Path to the source image (used only for logging).

        Returns:
            List[Dict]: Each item contains:
                - 'position': bounding box or polygon
                - 'texts': list of recognized text items within the stamp region
        """
        from PIL import ImageDraw
        import numpy as np

        verified_stamps = []

        for idx, stamp in enumerate(final_stamps):
            bbox = stamp['bbox']

            # Convert polygon to bounding box if needed
            if isinstance(bbox, np.ndarray):
                x_coords = [point[0][0] for point in bbox]
                y_coords = [point[0][1] for point in bbox]
                left, top, right, bottom = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                crop_box = (left, top, right, bottom)
            else:
                crop_box = tuple(bbox)  # assume [left, top, right, bottom]

            # Crop stamp region from image
            stamp_img = img.crop(crop_box)
            stamp_np = np.array(stamp_img)

            # Run OCR on cropped stamp
            ocr_results = self._fast_ocr(stamp_np)
            texts = []
            for text, conf, _ in ocr_results:
                if conf >= 0.5:
                    texts.append(text)

            verified_stamps.append({
                "position": bbox,
                "texts": texts
            })

        return verified_stamps

    def _verify_text_proximity(self, stamps: List[np.ndarray], image: np.ndarray) -> List[Dict]:
        import cv2
        proximity_ratio = 1.5
        keywords = ["公司", "有限", "集团"]
        verified = []

        for stamp in stamps:
            # stamp is a contour (np.ndarray), get bounding box
            x, y, w, h = cv2.boundingRect(stamp)
            x1, y1, x2, y2 = x, y, x + w, y + h

            roi_x1 = max(0, int(x1 - w * proximity_ratio))
            roi_y1 = max(0, int(y1 - h * proximity_ratio))
            roi_x2 = min(image.shape[1], int(x2 + w * proximity_ratio))
            roi_y2 = min(image.shape[0], int(y2 + h * proximity_ratio))

            roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
            ocr_results = self._fast_ocr(roi)

            related_texts = []
            for text, conf, pos in ocr_results:
                if conf < 0.6:
                    continue
                if any(keyword.lower() in text.lower() for keyword in keywords):
                    abs_pos = [[p[0] + roi_x1, p[1] + roi_y1] for p in pos]
                    related_texts.append({"text": text, "confidence": conf, "position": abs_pos})

            if related_texts:
                verified.append({
                    "bbox": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "related_texts": related_texts
                })

        return verified

    def _fast_ocr(self, image: np.ndarray):
        """
        Optimized OCR using PaddleOCR with error handling and robustness.
        Args:
            image (np.ndarray): Input image in BGR format.
        Returns:
            List[Tuple[str, float, List]]: List of (text, confidence, box) for each detected text.
        """
        try:
            # Resize if too large
            max_dim = 1024
            h, w = image.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)))

            # Run OCR
            results = self.ocr_engine.ocr(image)

            texts = []
            if not results or not isinstance(results, list):
                return []

            for result in results:
                if not result or not isinstance(result, list):
                    continue
                for line in result:
                    if not isinstance(line, list) or len(line) < 2:
                        continue
                    box = line[0]
                    text_info = line[1]
                    if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
                        continue
                    text, confidence = text_info[0], text_info[1]
                    if isinstance(text, str) and isinstance(confidence, (float, int)):
                        texts.append((text, confidence, box))

            return texts

        except Exception as e:
            print(f"Fast OCR failed: {e}")
            return []




if __name__ == '__main__':
    detector = StampDetector()
    results = detector._detect_stamps('C:/log/page_0001.png', OcrConfig())
    print(results)