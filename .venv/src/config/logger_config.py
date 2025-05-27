# logger_config.py
import logging
import logging.config
from pathlib import Path

# 配置字典（可自定义修改）
LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'filename': './app.log',
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5,
            'encoding': 'utf8'
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'standard',
            'filename': './error.log',
            'maxBytes': 5 * 1024 * 1024,  # 5MB
            'backupCount': 3,
            'encoding': 'utf8'
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file', 'error_file'],
            'level': 'DEBUG',
            'propagate': True
        },
        'http': {  # 示例：为特定模块设置独立配置
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}


class LoggerManager:
    """日志管理器单例类"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._configure_logging()
        return cls._instance

    @classmethod
    def _configure_logging(cls):
        """配置日志系统"""
        logging.config.dictConfig(LOG_CONFIG)
        # 创建基础logger（其他logger会继承其配置）
        logging.getLogger()

    @classmethod
    def get_logger(cls, name=None):
        """获取指定名称的logger实例

        Args:
            name: logger名称（默认为__name__）

        Returns:
            logging.Logger: 配置好的logger实例
        """
        if name is None:
            name = Path(__file__).stem  # 默认使用模块名
        return logging.getLogger(name)


# 便捷访问接口
def get_logger(name=None):
    """全局访问入口"""
    return LoggerManager.get_logger(name)


# 测试代码（实际使用时可以删除）
if __name__ == '__main__':
    # 测试不同logger
    main_logger = get_logger()
    http_logger = get_logger('http')

    main_logger.debug("这是主模块的调试信息")
    main_logger.info("程序启动成功")

    http_logger.warning("HTTP请求超时")
    http_logger.error("数据库连接失败")

    print("日志测试完成，请检查app.log和error.log文件")