class OcrConfig:
    LANGUAGES = ['chi_sim', 'eng']

    # 图像预处理参数
    PREPROCESS_STEPS = [
        'convert_grayscale',
        'enhance_contrast'
    ]
    VALID_FORMATS = ["png", "jpg", "jpeg", "tiff"]

    # PDF处理参数
    PDF_TO_IMAGE_DPI = 300
    PDF_TO_IMAGE_FORMAT = 'PNG'

    ENABLE_POST_PROCESS = False
    auto_clean_temp = False

    output_dir= 'c:/log'