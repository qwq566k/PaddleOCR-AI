import json
from datetime import datetime


class ReportChecker:
    def __init__(self, report_data):
        self.report = report_data
        self.results = {}

    def check_all(self):
        """执行所有检查项"""
        self.results = {
            "标题检查": self.check_title(),
            "资质标志和印章检查": self.check_qualification_marks(),
            "用章管理规定检查": self.check_seal_regulations(),
            "机构信息检查": self.check_organization_info(),
            "唯一性标识检查": self.check_unique_identifier(),
            "客户信息检查": self.check_client_info(),
            "日期信息检查": self.check_dates(),
            "签发人和单位检查": self.check_issuer_and_units()
        }
        return self.results

    def check_title(self):
        """检查1: 是否包括标题"""
        return "title" in self.report and bool(self.report["title"].strip())

    def check_qualification_marks(self):
        """检查2: 是否按要求加盖资质认定标志和检验检测专用章"""
        has_qualification_mark = self.report.get("has_qualification_mark", False)
        has_special_seal = self.report.get("has_special_seal", False)
        return has_qualification_mark and has_special_seal

    def check_seal_regulations(self):
        """检查3: 用章的使用是否有文件规定并按照执行"""
        has_regulations = self.report.get("has_seal_regulations", False)
        follows_regulations = self.report.get("follows_seal_regulations", False)
        return has_regulations and follows_regulations

    def check_organization_info(self):
        """检查4: 是否包括机构名称、地址和检测地点(如果不同)"""
        required = ["organization_name", "organization_address"]
        has_required = all(key in self.report for key in required)

        # 检查检测地点是否与机构地址不同时需要注明
        if "test_location" in self.report and self.report["test_location"] != self.report.get("organization_address",
                                                                                              ""):
            return has_required and bool(self.report["test_location"].strip())
        return has_required

    def check_unique_identifier(self):
        """检查5: 是否有唯一性标识"""
        return "unique_id" in self.report and bool(self.report["unique_id"].strip())

    def check_client_info(self):
        """检查6: 是否包括客户名称和联系信息"""
        required = ["client_name", "client_contact"]
        return all(key in self.report for key in required)

    def check_dates(self):
        """检查7: 是否包括检验检测日期，必要时注明接收/抽样日期"""
        has_test_date = "test_date" in self.report

        # 检查是否有重大影响日期需要注明
        if self.report.get("has_critical_dates", False):
            has_receipt_or_sampling_date = "receipt_date" in self.report or "sampling_date" in self.report
            return has_test_date and has_receipt_or_sampling_date
        return has_test_date

    def check_issuer_and_units(self):
        """检查8: 是否有签发人信息(授权签字人)和测量单位(适用时)"""
        has_issuer = "issuer_name" in self.report and self.report.get("is_authorized_signer", False)

        # 如果适用测量单位，则检查是否包含
        if self.report.get("requires_measurement_units", False):
            has_units = "measurement_units" in self.report
            return has_issuer and has_units
        return has_issuer


def load_report_data(file_path):
    """从JSON文件加载报告数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    # 示例使用
    report_data = {
        "title": "产品质量检验报告",
        "has_qualification_mark": True,
        "has_special_seal": True,
        "has_seal_regulations": True,
        "follows_seal_regulations": True,
        "organization_name": "XX检验检测机构",
        "organization_address": "XX市XX区XX路123号",
        "test_location": "XX市XX区XX路123号实验室",
        "unique_id": "REP-2023-001",
        "client_name": "ABC有限公司",
        "client_contact": "张经理 13800138000",
        "test_date": "2023-05-15",
        "has_critical_dates": True,
        "receipt_date": "2023-05-10",
        "issuer_name": "李四",
        "is_authorized_signer": True,
        "requires_measurement_units": True,
        "measurement_units": "mm/kg/s"
    }

    checker = ReportChecker(report_data)
    results = checker.check_all()

    print("检验检测报告合规性检查结果:")
    for i, (check_name, passed) in enumerate(results.items(), 1):
        status = "通过" if passed else "不通过"
        print(f"{i}. {check_name}: {status}")


if __name__ == "__main__":
    main()