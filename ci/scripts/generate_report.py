#!/usr/bin/env python3
"""
Enhanced CI Report Generator
生成易读、美观的CI测试报告
"""

import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import argparse

class CIReportGenerator:
    """生成美观的CI报告"""
    
    def __init__(self):
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'green': '\033[32m',
            'red': '\033[31m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'purple': '\033[35m',
            'cyan': '\033[36m'
        }
    
    def colorize(self, text: str, color: str) -> str:
        """给文本添加颜色"""
        return f"{self.colors.get(color, '')}{text}{self.colors['reset']}"
    
    def remove_ansi_codes(self, text: str) -> str:
        """移除ANSI颜色代码"""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)
    
    def parse_pytest_output(self, stdout: str) -> Dict[str, Any]:
        """解析pytest输出获取详细信息"""
        clean_output = self.remove_ansi_codes(stdout)
        
        result = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'deselected': 0,
            'errors': 0,
            'failed_tests': [],
            'error_tests': [],
            'execution_time': 0.0,
            'test_categories': {
                'import_tests': {'passed': 0, 'failed': 0},
                'cuda_kernel_tests': {'passed': 0, 'failed': 0},
                'model_conversion_tests': {'passed': 0, 'failed': 0}
            },
            'key_findings': []
        }
        
        lines = clean_output.split('\n')
        
        # 解析测试结果行
        for line in lines:
            if '::' in line and ('PASSED' in line or 'FAILED' in line or 'ERROR' in line):
                # 解析测试名称和分类
                parts = line.split('::')
                test_method = parts[-1].split()[0] if len(parts) > 0 else ""
                
                # 分类计数
                if 'test_imports.py' in line:
                    if 'PASSED' in line:
                        result['test_categories']['import_tests']['passed'] += 1
                        result['passed'] += 1
                    elif 'FAILED' in line:
                        result['test_categories']['import_tests']['failed'] += 1
                        result['failed'] += 1
                        result['failed_tests'].append(test_method)
                elif 'test_cuda_kernels.py' in line:
                    if 'PASSED' in line:
                        result['test_categories']['cuda_kernel_tests']['passed'] += 1
                        result['passed'] += 1
                    elif 'FAILED' in line:
                        result['test_categories']['cuda_kernel_tests']['failed'] += 1
                        result['failed'] += 1
                        result['failed_tests'].append(test_method)
                elif 'test_model_conversion.py' in line:
                    if 'PASSED' in line:
                        result['test_categories']['model_conversion_tests']['passed'] += 1
                        result['passed'] += 1
                    elif 'FAILED' in line:
                        result['test_categories']['model_conversion_tests']['failed'] += 1
                        result['failed'] += 1
                        result['failed_tests'].append(test_method)
                
                # 记录ERROR测试
                if 'ERROR' in line:
                    result['errors'] += 1
                    result['error_tests'].append(test_method)
        
        # 去重
        result['failed_tests'] = list(set(result['failed_tests']))
        result['error_tests'] = list(set(result['error_tests']))
        
        # 解析汇总行（优先使用汇总行的数据）
        for line in lines:
            if re.search(r'\d+\s+(failed|passed|error)', line) and 'in' in line:
                # 例如: "10 failed, 38 passed, 3 warnings, 1 error in 2.51s"
                failed_match = re.search(r'(\d+)\s+failed', line)
                passed_match = re.search(r'(\d+)\s+passed', line)
                error_match = re.search(r'(\d+)\s+error', line)
                time_match = re.search(r'in\s+([\d.]+)s', line)
                
                if failed_match:
                    result['failed'] = int(failed_match.group(1))
                if passed_match:
                    result['passed'] = int(passed_match.group(1))
                if error_match:
                    result['errors'] = int(error_match.group(1))
                if time_match:
                    result['execution_time'] = float(time_match.group(1))
                break
        
        result['total'] = result['passed'] + result['failed'] + result['errors']
        
        # 生成关键发现
        if result['test_categories']['import_tests']['failed'] == 0 and result['test_categories']['import_tests']['passed'] > 0:
            result['key_findings'].append("✅ 所有导入测试通过")
        elif result['test_categories']['import_tests']['failed'] > 0:
            result['key_findings'].append(f"❌ {result['test_categories']['import_tests']['failed']} 个导入测试失败")
            
        if result['test_categories']['cuda_kernel_tests']['failed'] > 0:
            result['key_findings'].append(f"⚠️ {result['test_categories']['cuda_kernel_tests']['failed']} 个CUDA内核测试失败")
        elif result['test_categories']['cuda_kernel_tests']['passed'] > 0:
            result['key_findings'].append(f"✅ {result['test_categories']['cuda_kernel_tests']['passed']} 个CUDA内核测试通过")
            
        if result['test_categories']['model_conversion_tests']['failed'] > 0:
            result['key_findings'].append(f"⚠️ {result['test_categories']['model_conversion_tests']['failed']} 个模型转换测试失败")
        elif result['test_categories']['model_conversion_tests']['passed'] > 0:
            result['key_findings'].append(f"✅ {result['test_categories']['model_conversion_tests']['passed']} 个模型转换测试通过")
        
        if result['errors'] > 0:
            result['key_findings'].append(f"💥 {result['errors']} 个测试出现错误")
        
        return result
    
    def generate_console_report(self, data: Dict[str, Any]) -> str:
        """生成控制台友好的报告"""
        report = []
        
        # 标题
        report.append("=" * 80)
        report.append(self.colorize("🧪 ST-BIF CI/CD 测试报告", 'bold'))
        report.append("=" * 80)
        
        # 基本信息
        timestamp = data.get('timestamp', 'N/A')
        if timestamp != 'N/A':
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                formatted_time = timestamp
        else:
            formatted_time = 'N/A'
        
        git_info = data.get('git_commit', {})
        
        report.append(f"📅 测试时间: {formatted_time}")
        report.append(f"🔗 Git分支: {git_info.get('branch', 'N/A')}")
        report.append(f"📦 Git提交: {git_info.get('commit', 'N/A')[:8]}")
        report.append("")
        
        # 总体结果
        summary = data.get('summary', {})
        total_suites = summary.get('total_test_suites', 0)
        passed_suites = summary.get('passed_test_suites', 0)
        failed_suites = summary.get('failed_test_suites', 0)
        overall_success = summary.get('overall_success', False)
        
        report.append("📊 总体结果")
        report.append("-" * 40)
        
        status_icon = "✅" if overall_success else "❌"
        status_text = self.colorize("通过", 'green') if overall_success else self.colorize("失败", 'red')
        report.append(f"{status_icon} 整体状态: {status_text}")
        
        if total_suites > 0:
            success_rate = (passed_suites / total_suites) * 100
            report.append(f"📈 成功率: {success_rate:.1f}% ({passed_suites}/{total_suites})")
        
        report.append("")
        
        # 详细测试套件结果
        report.append("🔍 测试套件详情")
        report.append("-" * 40)
        
        test_runs = data.get('test_runs', {})
        total_tests = 0
        total_passed_tests = 0
        total_failed_tests = 0
        
        for suite_name, suite_data in test_runs.items():
            success = suite_data.get('success', False)
            exec_time = suite_data.get('execution_time', 0)
            
            # 解析pytest输出
            pytest_info = self.parse_pytest_output(suite_data.get('stdout', ''))
            
            # 图标和状态
            if success:
                icon = "✅"
                status_color = 'green'
                status_text = "通过"
            else:
                icon = "❌"
                status_color = 'red'
                status_text = "失败"
            
            report.append(f"{icon} {self.colorize(suite_name, 'bold')}")
            report.append(f"   状态: {self.colorize(status_text, status_color)}")
            report.append(f"   耗时: {exec_time:.1f}秒")
            
            if pytest_info['total'] > 0:
                report.append(f"   测试: {pytest_info['passed']}通过, {pytest_info['failed']}失败")
                if pytest_info['errors'] > 0:
                    report.append(f"         {pytest_info['errors']}错误")
                if pytest_info['skipped'] > 0:
                    report.append(f"         {pytest_info['skipped']}跳过")
                
                # 显示测试分类统计
                categories = pytest_info.get('test_categories', {})
                for cat_name, cat_data in categories.items():
                    if cat_data['passed'] > 0 or cat_data['failed'] > 0:
                        cat_display = {
                            'import_tests': '导入',
                            'cuda_kernel_tests': 'CUDA内核', 
                            'model_conversion_tests': '模型转换'
                        }.get(cat_name, cat_name)
                        report.append(f"   {cat_display}: {cat_data['passed']}通过, {cat_data['failed']}失败")
                
                total_tests += pytest_info['total']
                total_passed_tests += pytest_info['passed']
                total_failed_tests += pytest_info['failed']
                
                # 显示失败的测试
                if pytest_info['failed_tests']:
                    report.append(f"   {self.colorize('失败测试:', 'red')}")
                    for failed_test in pytest_info['failed_tests'][:3]:  # 只显示前3个
                        report.append(f"     • {failed_test}")
                    if len(pytest_info['failed_tests']) > 3:
                        report.append(f"     ... 还有 {len(pytest_info['failed_tests']) - 3} 个失败测试")
                
                # 显示错误的测试
                if pytest_info['error_tests']:
                    report.append(f"   {self.colorize('错误测试:', 'red')}")
                    for error_test in pytest_info['error_tests'][:2]:  # 只显示前2个
                        report.append(f"     • {error_test}")
                    if len(pytest_info['error_tests']) > 2:
                        report.append(f"     ... 还有 {len(pytest_info['error_tests']) - 2} 个错误测试")
            
            # 错误信息
            if not success and 'stderr' in suite_data and suite_data['stderr']:
                error_msg = suite_data['stderr'][:200]
                report.append(f"   {self.colorize('错误:', 'red')} {error_msg}")
            
            report.append("")
        
        # 测试统计汇总
        if total_tests > 0:
            report.append("📈 测试统计汇总")
            report.append("-" * 40)
            report.append(f"总测试数: {total_tests}")
            report.append(f"通过: {self.colorize(str(total_passed_tests), 'green')}")
            report.append(f"失败: {self.colorize(str(total_failed_tests), 'red')}")
            if total_tests > 0:
                pass_rate = (total_passed_tests / total_tests) * 100
                color = 'green' if pass_rate >= 90 else 'yellow' if pass_rate >= 70 else 'red'
                report.append(f"通过率: {self.colorize(f'{pass_rate:.1f}%', color)}")
            report.append("")
        
        # 关键发现
        report.append("🔎 关键发现")
        report.append("-" * 40)
        
        key_findings = []
        
        # 从所有测试套件中汇总关键发现
        for suite_name, suite_data in test_runs.items():
            pytest_info = self.parse_pytest_output(suite_data.get('stdout', ''))
            key_findings.extend(pytest_info.get('key_findings', []))
        
        # 去重并按重要性排序
        unique_findings = list(dict.fromkeys(key_findings))  # 保持顺序的去重
        
        if not unique_findings:
            unique_findings.append("ℹ️  基础验证完成")
        
        for finding in unique_findings:
            report.append(f"  {finding}")
        
        # 额外分析
        cuda_issues = [f for f in unique_findings if 'CUDA' in f and ('失败' in f or '错误' in f)]
        if cuda_issues:
            report.append("  🚨 检测到CUDA相关问题，建议检查GPU环境")
        
        report.append("")
        
        # 建议和后续步骤
        if not overall_success:
            report.append("🚀 建议的后续步骤")
            report.append("-" * 40)
            
            suggestions = []
            
            # 基于失败的测试套件给出建议
            for suite_name, suite_data in test_runs.items():
                if not suite_data.get('success'):
                    if 'CUDA' in suite_name:
                        suggestions.append("🔧 检查CUDA环境和GPU可用性")
                        suggestions.append("📊 运行详细的CUDA内核分析")
                    elif 'Import' in suite_name:
                        suggestions.append("📦 检查Python环境和依赖安装")
                    elif 'Model' in suite_name:
                        suggestions.append("🧠 检查模型文件和转换流水线")
            
            if not suggestions:
                suggestions.append("🔍 查看详细日志文件获取更多信息")
                suggestions.append("🧪 运行特定失败的测试以获得更多细节")
            
            for suggestion in suggestions[:3]:  # 最多显示3个建议
                report.append(f"  {suggestion}")
            
            report.append("")
        
        # 结尾
        report.append("=" * 80)
        current_time = datetime.now().strftime('%H:%M:%S')
        report.append(f"📋 报告生成时间: {current_time}")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def generate_html_report(self, data: Dict[str, Any]) -> str:
        """生成HTML格式报告"""
        html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ST-BIF CI/CD 测试报告</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .content {
            padding: 30px;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .summary-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #3498db;
        }
        .summary-card.success {
            border-left-color: #27ae60;
        }
        .summary-card.failed {
            border-left-color: #e74c3c;
        }
        .summary-card h3 {
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 1.1em;
        }
        .summary-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        .summary-card.success .value {
            color: #27ae60;
        }
        .summary-card.failed .value {
            color: #e74c3c;
        }
        .test-suite {
            background: #f8f9fa;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
        }
        .test-suite-header {
            padding: 20px;
            background: white;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .test-suite-header h3 {
            margin: 0;
            color: #2c3e50;
        }
        .status-badge {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }
        .status-badge.success {
            background: #d4edda;
            color: #155724;
        }
        .status-badge.failed {
            background: #f8d7da;
            color: #721c24;
        }
        .test-details {
            padding: 20px;
            display: none;
        }
        .test-details.show {
            display: block;
        }
        .test-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }
        .stat-item {
            text-align: center;
            padding: 10px;
            background: white;
            border-radius: 5px;
        }
        .stat-item .label {
            font-size: 0.9em;
            color: #6c757d;
        }
        .stat-item .value {
            font-size: 1.2em;
            font-weight: bold;
            color: #2c3e50;
        }
        .failed-tests {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin-top: 15px;
        }
        .failed-tests h4 {
            margin: 0 0 10px 0;
            color: #856404;
        }
        .failed-tests ul {
            margin: 0;
            padding-left: 20px;
        }
        .toggle-btn {
            background: none;
            border: none;
            color: #3498db;
            cursor: pointer;
            padding: 5px;
            border-radius: 3px;
        }
        .toggle-btn:hover {
            background: #e9ecef;
        }
        .footer {
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            color: #6c757d;
            border-top: 1px solid #e9ecef;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 ST-BIF CI/CD 测试报告</h1>
            <p>自动化测试和质量保证系统</p>
        </div>
        <div class="content">
"""
        
        # 基本信息和汇总
        summary = data.get('summary', {})
        timestamp = data.get('timestamp', 'N/A')
        git_info = data.get('git_commit', {})
        
        # 格式化时间
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            formatted_time = timestamp
        
        # 汇总卡片
        total_suites = summary.get('total_test_suites', 0)
        passed_suites = summary.get('passed_test_suites', 0)
        failed_suites = summary.get('failed_test_suites', 0)
        overall_success = summary.get('overall_success', False)
        
        success_rate = (passed_suites / total_suites * 100) if total_suites > 0 else 0
        
        html += f"""
            <div class="summary">
                <div class="summary-card {'success' if overall_success else 'failed'}">
                    <h3>整体状态</h3>
                    <div class="value">{'✅ 通过' if overall_success else '❌ 失败'}</div>
                </div>
                <div class="summary-card">
                    <h3>测试套件</h3>
                    <div class="value">{passed_suites}/{total_suites}</div>
                </div>
                <div class="summary-card {'success' if success_rate >= 80 else 'failed'}">
                    <h3>成功率</h3>
                    <div class="value">{success_rate:.1f}%</div>
                </div>
                <div class="summary-card">
                    <h3>执行时间</h3>
                    <div class="value">{formatted_time}</div>
                </div>
            </div>
            
            <p><strong>Git分支:</strong> {git_info.get('branch', 'N/A')}</p>
            <p><strong>Git提交:</strong> {git_info.get('commit', 'N/A')[:12]}</p>
"""
        
        # 测试套件详情
        test_runs = data.get('test_runs', {})
        for i, (suite_name, suite_data) in enumerate(test_runs.items()):
            success = suite_data.get('success', False)
            exec_time = suite_data.get('execution_time', 0)
            
            pytest_info = self.parse_pytest_output(suite_data.get('stdout', ''))
            
            html += f"""
            <div class="test-suite">
                <div class="test-suite-header">
                    <h3>{suite_name}</h3>
                    <div>
                        <span class="status-badge {'success' if success else 'failed'}">
                            {'通过' if success else '失败'}
                        </span>
                        <button class="toggle-btn" onclick="toggleDetails({i})">详情</button>
                    </div>
                </div>
                <div class="test-details" id="details-{i}">
                    <div class="test-stats">
                        <div class="stat-item">
                            <div class="label">执行时间</div>
                            <div class="value">{exec_time:.1f}s</div>
                        </div>
            """
            
            if pytest_info['total'] > 0:
                html += f"""
                        <div class="stat-item">
                            <div class="label">总测试</div>
                            <div class="value">{pytest_info['total']}</div>
                        </div>
                        <div class="stat-item">
                            <div class="label">通过</div>
                            <div class="value">{pytest_info['passed']}</div>
                        </div>
                        <div class="stat-item">
                            <div class="label">失败</div>
                            <div class="value">{pytest_info['failed']}</div>
                        </div>
                """
                
                if pytest_info['failed_tests']:
                    html += """
                    </div>
                    <div class="failed-tests">
                        <h4>失败的测试</h4>
                        <ul>
                    """
                    for failed_test in pytest_info['failed_tests'][:10]:
                        html += f"<li>{failed_test}</li>"
                    html += "</ul></div>"
                else:
                    html += "</div>"
            else:
                html += "</div>"
            
            html += "</div></div>"
        
        # 结尾
        html += """
        </div>
        <div class="footer">
            <p>由 ST-BIF CI/CD 系统自动生成</p>
        </div>
    </div>
    
    <script>
        function toggleDetails(index) {
            const details = document.getElementById('details-' + index);
            if (details.classList.contains('show')) {
                details.classList.remove('show');
            } else {
                details.classList.add('show');
            }
        }
    </script>
</body>
</html>
"""
        return html
    
    def generate_markdown_report(self, data: Dict[str, Any]) -> str:
        """生成Markdown格式报告"""
        md = []
        
        # 标题
        md.append("# 🧪 ST-BIF CI/CD 测试报告")
        md.append("")
        
        # 基本信息
        timestamp = data.get('timestamp', 'N/A')
        git_info = data.get('git_commit', {})
        
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            formatted_time = timestamp
        
        md.append("## 📋 基本信息")
        md.append("")
        md.append(f"- **测试时间**: {formatted_time}")
        md.append(f"- **Git分支**: {git_info.get('branch', 'N/A')}")
        md.append(f"- **Git提交**: `{git_info.get('commit', 'N/A')[:12]}`")
        md.append("")
        
        # 总体结果
        summary = data.get('summary', {})
        overall_success = summary.get('overall_success', False)
        
        status_emoji = "✅" if overall_success else "❌"
        md.append(f"## {status_emoji} 总体结果")
        md.append("")
        
        total_suites = summary.get('total_test_suites', 0)
        passed_suites = summary.get('passed_test_suites', 0)
        
        if total_suites > 0:
            success_rate = (passed_suites / total_suites) * 100
            md.append(f"- **状态**: {'通过' if overall_success else '失败'}")
            md.append(f"- **成功率**: {success_rate:.1f}% ({passed_suites}/{total_suites})")
            md.append("")
        
        # 测试套件详情
        md.append("## 🔍 测试套件详情")
        md.append("")
        
        test_runs = data.get('test_runs', {})
        for suite_name, suite_data in test_runs.items():
            success = suite_data.get('success', False)
            exec_time = suite_data.get('execution_time', 0)
            
            icon = "✅" if success else "❌"
            md.append(f"### {icon} {suite_name}")
            md.append("")
            
            pytest_info = self.parse_pytest_output(suite_data.get('stdout', ''))
            
            md.append(f"- **状态**: {'通过' if success else '失败'}")
            md.append(f"- **执行时间**: {exec_time:.1f}秒")
            
            if pytest_info['total'] > 0:
                md.append(f"- **测试统计**: {pytest_info['passed']}通过, {pytest_info['failed']}失败")
                
                if pytest_info['failed_tests']:
                    md.append("- **失败测试**:")
                    for failed_test in pytest_info['failed_tests'][:5]:
                        md.append(f"  - `{failed_test}`")
            
            md.append("")
        
        return "\n".join(md)

def main():
    parser = argparse.ArgumentParser(description="生成CI测试报告")
    parser.add_argument("--input", "-i", 
                       default="ci/results/latest/pytest_ci_results.json",
                       help="输入的JSON结果文件")
    parser.add_argument("--format", "-f", 
                       choices=['console', 'html', 'markdown', 'all'],
                       default='console',
                       help="报告格式")
    parser.add_argument("--output", "-o",
                       help="输出文件路径（默认输出到控制台）")
    
    args = parser.parse_args()
    
    # 读取数据
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"错误：找不到文件 {input_file}")
        return 1
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"错误：无法读取JSON文件 - {e}")
        return 1
    
    # 生成报告
    generator = CIReportGenerator()
    
    if args.format == 'console' or args.format == 'all':
        console_report = generator.generate_console_report(data)
        if args.output and args.format == 'console':
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(console_report)
            print(f"控制台报告已保存到: {args.output}")
        else:
            print(console_report)
    
    if args.format == 'html' or args.format == 'all':
        html_report = generator.generate_html_report(data)
        html_file = args.output if args.output else "ci/results/latest/report.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_report)
        print(f"HTML报告已保存到: {html_file}")
    
    if args.format == 'markdown' or args.format == 'all':
        md_report = generator.generate_markdown_report(data)
        md_file = args.output if args.output else "ci/results/latest/report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_report)
        print(f"Markdown报告已保存到: {md_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())