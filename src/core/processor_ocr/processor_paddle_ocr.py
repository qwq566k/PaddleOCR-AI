
"""处理PDF文档"""
import os
from typing import List, Tuple, Dict

import numpy as np
import paddle
from PIL import Image
from paddleocr import PaddleOCR

from config.logger_config import get_logger
from config.ocr_settings import OcrConfig
from core.preprocessor.pdf_to_img import PdfToImgWrapper
from core.processor_stamp_detector.StampDetector import StampDetector


class PaddleOCRWrapper:
    def __init__(self, lang='ch'):
        self.logger = get_logger('PaddleOCRWrapper')
        self.ocr = PaddleOCR(
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False)
        self.pdftoimg = PdfToImgWrapper()
        self.stampDetector = StampDetector()

    def process_pdf(self, pdf_path: str, config) -> dict:
        # 获取绝对父路径
        img_dir = os.path.dirname(os.path.abspath(pdf_path))
        filename = os.path.basename(pdf_path)
        img_dir=os.path.join(img_dir,os.path.splitext(filename)[0])
        image_paths = self.pdftoimg.pdf_to_images(pdf_path, img_dir, config)
        self.extract_text(image_paths)


    def extract_text(self, image_paths: List[str]):
        """
        处理硬盘上的图像文件
        :param image_path: 图像文件路径列表
        :return: OCR结果列表，格式为[(text, confidence, position), ...]
        """
        for image_path in image_paths:
            if not os.path.exists(image_path):
                self.logger.error(f"图像文件不存在: {image_path}")
                continue

            try:
                # 对图像执行 OCR 推理
                result = self.ocr.predict(
                    input=image_path)

                # 可视化结果并保存 json 结果
                for res in result:
                    res.print()
                    res.save_to_img("output")
                    res.save_to_json("output")

            except Exception as e:
                self.logger.error(f"PDF{image_path} ocr失败: {str(e)}")

    def init(img_dir: str):
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(img_dir + '/done', exist_ok=True)

if __name__ == '__main__':
    # paddle.utils.run_check()
    ocr = PaddleOCRWrapper()
    ocr._detect_stamps('D:\夸克网盘\1000 X工作室\02 项目- 南京CNAS智能审核助手\03 结果样本\整改材料.pdf', OcrConfig())
    # ocr.process_pdf("C:/log/评审不符合项整改材料.pdf", OcrConfig())