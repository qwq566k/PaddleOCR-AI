"""
签章文本分析工具 - 完整注释版
功能：从OCR结果JSON中提取关键词相关文本及关联签章位置
"""

import json
from typing import List, Dict, Tuple


class StampTextAnalyzer:
    def __init__(self, json_path: str):
        """分析器初始化
        Args:
            json_path: OCR结果JSON文件路径
        """
        self.json_path = json_path  # 原始数据文件路径
        self.data = self._load_and_validate_data()  # 加载并验证后的数据

    def _load_and_validate_data(self) -> dict:
        """加载并验证JSON数据结构
        Returns:
            验证后的数据字典
        Raises:
            ValueError: 数据格式错误时抛出
            RuntimeError: 文件加载失败时抛出
        """
        try:
            # 打开并读取JSON文件
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 必须包含的关键字段验证
            required_fields = ['rec_texts', 'rec_boxes', 'dt_polys', 'rec_scores']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"关键字段缺失: {field}")

            # 数据长度一致性验证
            if (len(data['rec_texts']) != len(data['rec_boxes']) or
                    len(data['rec_texts']) != len(data['rec_scores'])):
                raise ValueError("文本/坐标/置信度数据长度不一致")

            return data
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON解析失败: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"文件加载失败: {str(e)}")

    def find_keyword_matches(self,
                             keywords: List[str],
                             min_confidence: float = 0.5) -> List[Dict]:
        """查找包含关键词的文本及其位置
        Args:
            keywords: 待匹配关键词列表
            min_confidence: 最低置信度阈值(默认0.5)
        Returns:
            包含匹配结果的字典列表，结构：
            [{
                'text': 识别文本,
                'confidence': 置信度,
                'bounding_box': 坐标信息,
                'matched_keywords': 匹配到的关键词
            }]
        """
        results = []
        # 遍历所有识别结果
        for idx, (text, score) in enumerate(zip(self.data['rec_texts'],
                                                self.data['rec_scores'])):
            # 过滤空文本和低置信度结果
            if not text.strip() or score < min_confidence:
                continue

            # 检查包含哪些关键词
            matched_keywords = [kw for kw in keywords if kw in text]
            if matched_keywords:
                # 提取对应坐标框
                x1, y1, x2, y2 = self.data['rec_boxes'][idx]
                # 构建结果字典
                results.append({
                    'text': text,
                    'confidence': round(score, 4),  # 保留4位小数
                    'bounding_box': {
                        'left_top': (x1, y1),
                        'right_bottom': (x2, y2),
                        'center': ((x1 + x2) / 2, (y1 + y2) / 2)  # 计算中心点
                    },
                    'matched_keywords': matched_keywords
                })
        return results

    def _calculate_polygon_center(self,
                                  polygon: List[List[int]]) -> Tuple[float, float]:
        """计算多边形几何中心
        Args:
            polygon: 多边形坐标点列表 [[x1,y1], [x2,y2]...]
        Returns:
            (center_x, center_y) 中心坐标
        """
        # 提取所有x坐标和y坐标
        x_coords = [point[0] for point in polygon]
        y_coords = [point[1] for point in polygon]
        # 计算平均值得到中心点
        return sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)

    def find_related_stamps(self,
                            text_results: List[Dict],
                            max_distance: float = 200.0) -> List[Dict]:
        """查找与文本关联的签章
        Args:
            text_results: find_keyword_matches的输出结果
            max_distance: 最大关联距离(单位：像素)
        Returns:
            签章关联信息列表，结构：
            [{
                'stamp_polygon': 签章坐标,
                'stamp_center': 签章中心,
                'nearest_text': 最近文本,
                'distance': 距离(像素)
            }]
        """
        stamp_data = []
        # 遍历所有检测到的签章
        for poly in self.data['dt_polys']:
            # 计算当前签章中心点
            stamp_center = self._calculate_polygon_center(poly)

            # 初始化最近文本信息
            nearest_text = None
            min_distance = float('inf')  # 初始设为极大值

            # 遍历所有匹配文本
            for text_info in text_results:
                text_center = text_info['bounding_box']['center']
                # 计算欧氏距离
                distance = ((stamp_center[0] - text_center[0]) ** 2 +
                            (stamp_center[1] - text_center[1]) ** 2) ** 0.5

                # 更新最近文本
                if distance < min_distance and distance <= max_distance:
                    min_distance = distance
                    nearest_text = text_info['text']

            # 记录签章信息
            stamp_data.append({
                'stamp_polygon': poly,
                'stamp_center': stamp_center,
                'nearest_text': nearest_text,
                'distance': round(min_distance, 2) if nearest_text else None
            })
        return stamp_data

    def generate_report(self, keywords: List[str]) -> Dict:
        """生成完整分析报告
        Args:
            keywords: 搜索关键词列表
        Returns:
            结构化报告字典，包含：
            - metadata: 元数据
            - text_results: 文本匹配结果
            - stamp_relations: 签章关联结果
        """
        # 获取关键词匹配结果
        text_matches = self.find_keyword_matches(keywords)
        # 获取签章关联信息
        stamp_relations = self.find_related_stamps(text_matches)

        # 构建报告结构
        return {
            'metadata': {
                'source_file': self.json_path,
                'keywords': keywords,
                'text_count': len(self.data['rec_texts']),  # 总文本数量
                'stamp_count': len(self.data['dt_polys'])  # 总签章数量
            },
            'text_results': text_matches,
            'stamp_relations': stamp_relations
        }


def save_report(report: Dict, output_path: str = "stamp_analysis_report.json"):
    """保存分析报告到文件
    Args:
        report: generate_report生成的报告字典
        output_path: 输出文件路径(默认当前目录)
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f,
                  ensure_ascii=False,  # 保留中文
                  indent=2)  # 缩进美化
    print(f"分析报告已保存至: {output_path}")


if __name__ == "__main__":
    # ---------------------------
    # 使用示例
    # ---------------------------

    # 初始化分析器(替换为实际文件路径)
    analyzer = StampTextAnalyzer("output/page_0004_res.json")

    # 设置搜索关键词(根据业务需求调整)
    search_keywords = ["公司", "章", "批准", "编制", "专用"]

    # 生成分析报告
    analysis_report = analyzer.generate_report(search_keywords)

    # 控制台输出关键信息
    print("\n=== 关键词匹配结果 ===")
    for idx, text in enumerate(analysis_report['text_results'], 1):
        print(f"{idx}. 文本内容: {text['text']}")
        print(f"   匹配关键词: {text['matched_keywords']}")
        print(f"   置信度: {text['confidence']:.2%}")
        print(f"   中心坐标: {text['bounding_box']['center']}\n")

    print("\n=== 签章关联结果 ===")
    for idx, stamp in enumerate(analysis_report['stamp_relations'], 1):
        if stamp['nearest_text']:
            print(f"{idx}. 签章中心: ({stamp['stamp_center'][0]:.1f}, {stamp['stamp_center'][1]:.1f})")
            print(f"   最近文本: {stamp['nearest_text']}")
            print(f"   距离: {stamp['distance']} 像素\n")

    # 保存完整报告
    # save_report(analysis_report)