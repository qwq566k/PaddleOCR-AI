# 必需依赖：
# PyMuPDF (用于PDF解析) - 安装命令：pip install pymupdf
# Pillow (用于图像处理) - 安装命令：pip install pillow
import os
import shutil
import logging
from typing import List

import fitz
from PIL import Image  # 图像处理模块
import io  # 用于字节流操作

from config.logger_config import get_logger

class PdfToImgWrapper:
    def __init__(self):
        self.logger = get_logger('pdf_to_img')

    def pdf_to_images(self, pdf_path: str, img_dir: str, config) -> List[str]:
        """
        将PDF转换为高质量图像文件（适配PyMuPDF 1.26.0+）

        参数：
        pdf_path (str): PDF文件路径
        img_dir (str): 图像输出目录
        config (object): 配置对象，需包含：
            - PDF_TO_IMAGE_DPI (int): 输出分辨率（默认300）
            - PDF_TO_IMAGE_FORMAT (str): 图像格式（PNG/JPEG等）

        返回：
        List[str]: 生成的图像文件路径列表，按页码排序

        异常：
        FileNotFoundError: 输入文件不存在时抛出
        RuntimeError: PDF处理失败时抛出
        """
        # 参数校验前置
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        if not os.path.exists(img_dir):
            os.makedirs(img_dir, exist_ok=True)

        doc = None
        image_paths = []

        try:
            doc = fitz.open(pdf_path)

            for page_num, page in enumerate(doc, start=1):
                output_path = self._process_page(page_num, page, img_dir, config)
                if output_path:
                    image_paths.append(output_path)

            return image_paths
        except Exception as e:
            self.logger.error(f"PDF处理失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"PDF处理失败: {str(e)}") from e
        finally:
            if doc:
                doc.close()

    def _process_page(self, page_num: int, page, img_dir: str, config) -> str:
        """处理单个PDF页面（OCR优化版）"""
        filename = self.generate_image_filename(page_num, config)
        output_path = os.path.join(img_dir, filename)

        # 存在性检查保持跳过逻辑
        if os.path.exists(output_path):
            self.logger.debug(f"跳过已存在页面: {output_path}")
            return output_path

        try:
            # 优化图像生成参数
            pix = page.get_pixmap(
                dpi=max(config.PDF_TO_IMAGE_DPI, 300),  # OCR最低300DPI保障
                colorspace="rgb",  # 保留彩色信息用于签章识别
                annots=True,  # 包含注释（可能含签章元素）
                alpha=False,  # 禁用透明通道节省空间
                # antialias=True,  # 启用抗锯齿提升文字边缘质量
                clip=page.rect  # 完整页面区域
            )


            pix.save(output_path)

            # 后续处理建议
            if config.ENABLE_POST_PROCESS:
                self._post_process_image(output_path)

            return output_path
        except Exception as e:
            self.logger.error(f"页面{page_num}处理失败: {str(e)}", exc_info=True)
            return None

    def _post_process_image(self, image_path: str):
        """图像后处理（可选）"""
        from PIL import Image, ImageEnhance

        # 示例：对比度增强
        with Image.open(image_path) as img:
            enhancer = ImageEnhance.Contrast(img)
            enhanced = enhancer.enhance(1.2)  # 增强20%
            enhanced.save(image_path, quality=100)

    def generate_image_filename(self, page_num: int, config) -> str:
        """生成标准化图像文件名"""
        # 验证文件格式合法性

        fmt = config.PDF_TO_IMAGE_FORMAT.lower()
        if fmt not in config.VALID_FORMATS:
            raise ValueError(f"不支持的图像格式: {fmt}，可选格式: {config.VALID_FORMATS}")

        # 自动校正扩展名（如将jpeg转为jpg）
        ext = "jpg" if fmt in ["jpg", "jpeg"] else fmt
        return f"page_{page_num:04d}.{ext}"

