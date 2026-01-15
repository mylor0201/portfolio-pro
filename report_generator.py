"""
Client Report Generator
Tạo báo cáo PDF chuyên nghiệp cho khách hàng

Sử dụng: 
    from report_generator import ReportGenerator
    report = ReportGenerator(client_name, holdings, start_date, end_date)
    report.generate('output.pdf')
"""

from fpdf import FPDF
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For non-GUI backend
import tempfile
import os

from analysis_engine import PortfolioAnalyzer, PortfolioMetrics, StockMetrics


def normalize_vietnamese(text: str) -> str:
    """
    Convert Vietnamese text to ASCII-safe format by removing diacritics
    This prevents font encoding errors in PDF generation
    """
    replacements = {
        'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
        'đ': 'd',
        'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
        'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
        'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
        'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
        'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
        # Uppercase
        'À': 'A', 'Á': 'A', 'Ả': 'A', 'Ã': 'A', 'Ạ': 'A',
        'Ă': 'A', 'Ằ': 'A', 'Ắ': 'A', 'Ẳ': 'A', 'Ẵ': 'A', 'Ặ': 'A',
        'Â': 'A', 'Ầ': 'A', 'Ấ': 'A', 'Ẩ': 'A', 'Ẫ': 'A', 'Ậ': 'A',
        'Đ': 'D',
        'È': 'E', 'É': 'E', 'Ẻ': 'E', 'Ẽ': 'E', 'Ẹ': 'E',
        'Ê': 'E', 'Ề': 'E', 'Ế': 'E', 'Ể': 'E', 'Ễ': 'E', 'Ệ': 'E',
        'Ì': 'I', 'Í': 'I', 'Ỉ': 'I', 'Ĩ': 'I', 'Ị': 'I',
        'Ò': 'O', 'Ó': 'O', 'Ỏ': 'O', 'Õ': 'O', 'Ọ': 'O',
        'Ô': 'O', 'Ồ': 'O', 'Ố': 'O', 'Ổ': 'O', 'Ỗ': 'O', 'Ộ': 'O',
        'Ơ': 'O', 'Ờ': 'O', 'Ớ': 'O', 'Ở': 'O', 'Ỡ': 'O', 'Ợ': 'O',
        'Ù': 'U', 'Ú': 'U', 'Ủ': 'U', 'Ũ': 'U', 'Ụ': 'U',
        'Ư': 'U', 'Ừ': 'U', 'Ứ': 'U', 'Ử': 'U', 'Ữ': 'U', 'Ự': 'U',
        'Ỳ': 'Y', 'Ý': 'Y', 'Ỷ': 'Y', 'Ỹ': 'Y', 'Ỵ': 'Y',
        # Special characters
        '•': '-', '–': '-', '—': '-', '…': '...', 
        '"': '"', '"': '"', ''': "'", ''': "'",
        '₫': 'VND', '€': 'EUR', '£': 'GBP', '¥': 'YEN',
    }
    
    result = text
    for viet_char, ascii_char in replacements.items():
        result = result.replace(viet_char, ascii_char)
    
    return result


class ReportGenerator:
    """
    Tạo báo cáo PDF chuyên nghiệp cho khách hàng
    """
    
    def __init__(
        self,
        client_name: str,
        holdings: Dict[str, float],
        start_date: str = None,
        end_date: str = None,
        broker_name: str = "Tran Quang Huy - SSI Securities (Broker ID: 2537)",
        broker_phone: str = "0902571858",
        broker_email: str = "huytq2@ssi.com.vn",
        company_name: str = "SSI Securities Corporation"
    ):
        """
        Parameters:
        - client_name: Tên khách hàng
        - holdings: Dictionary {symbol: weight}
        - start_date, end_date: Khoảng thời gian phân tích
        - broker_*: Thông tin môi giới để footer
        """
        self.client_name = client_name
        self.holdings = holdings
        self.broker_name = broker_name
        self.broker_phone = broker_phone
        self.broker_email = broker_email
        self.company_name = company_name
        
        # Run analysis
        self.analyzer = PortfolioAnalyzer(holdings, start_date, end_date)
        self.metrics = self.analyzer.get_portfolio_metrics()
        self.stock_metrics = self.analyzer.get_stock_metrics()
        self.sector_alloc = self.analyzer.get_sector_allocation()
        self.risk_assessment = self.analyzer.get_risk_assessment()
        self.cumulative_returns = self.analyzer.get_cumulative_returns()
        self.monthly_returns = self.analyzer.get_monthly_returns()
        
        self.report_date = datetime.now().strftime('%d/%m/%Y')
        self.period = f"{self.analyzer.start_date} đến {self.analyzer.end_date}"
        
        # Temp files for charts
        self.temp_files = []
    
    def _create_performance_chart(self) -> str:
        """Tạo biểu đồ hiệu suất và return đường dẫn file"""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        df = self.cumulative_returns
        ax.plot(df['date'], df['portfolio'], label='Danh mục', color='#667eea', linewidth=2)
        ax.plot(df['date'], df['benchmark'], label='VN-Index', color='#f5a623', linewidth=2, linestyle='--')
        
        ax.set_xlabel('Ngày')
        ax.set_ylabel('Lợi nhuận (%)')
        ax.set_title('Hiệu Suất Danh Mục vs VN-Index')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save to temp file
        temp_path = tempfile.mktemp(suffix='.png')
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.temp_files.append(temp_path)
        return temp_path
    
    def _create_sector_chart(self) -> str:
        """Tạo biểu đồ phân bổ ngành"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sectors = list(self.sector_alloc.keys())
        weights = list(self.sector_alloc.values())
        colors = plt.cm.Set2(np.linspace(0, 1, len(sectors)))
        
        wedges, texts, autotexts = ax.pie(
            weights, 
            labels=sectors, 
            autopct='%1.1f%%',
            colors=colors,
            pctdistance=0.75
        )
        
        # Draw circle for donut chart
        centre_circle = plt.Circle((0, 0), 0.50, fc='white')
        ax.add_patch(centre_circle)
        
        ax.set_title('Phân Bổ Theo Ngành')
        plt.tight_layout()
        
        temp_path = tempfile.mktemp(suffix='.png')
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.temp_files.append(temp_path)
        return temp_path
    
    def _create_monthly_chart(self) -> str:
        """Tạo biểu đồ lợi nhuận theo tháng"""
        fig, ax = plt.subplots(figsize=(10, 4))
        
        df = self.monthly_returns
        colors = ['#00c853' if x >= 0 else '#ff5252' for x in df['return']]
        
        ax.bar(df['month'], df['return'], color=colors)
        ax.set_xlabel('Tháng')
        ax.set_ylabel('Lợi nhuận (%)')
        ax.set_title('Lợi Nhuận Theo Tháng')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        temp_path = tempfile.mktemp(suffix='.png')
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.temp_files.append(temp_path)
        return temp_path
    
    def generate(self, output_path: str) -> str:
        """
        Tạo báo cáo PDF
        
        Parameters:
        - output_path: Đường dẫn file output
        
        Returns:
        - Đường dẫn file PDF đã tạo
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Add Vietnamese font (if available, otherwise use default)
        try:
            # You can add Vietnamese font here
            # pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
            pass
        except:
            pass
        
        # ============== TRANG 1: TỔNG QUAN ==============
        pdf.add_page()
        
        # Header
        pdf.set_font('Helvetica', 'B', 24)
        pdf.set_text_color(102, 126, 234)  # #667eea
        pdf.cell(0, 15, normalize_vietnamese('BAO CAO DANH MUC DAU TU'), ln=True, align='C')
        
        pdf.set_font('Helvetica', '', 12)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 8, normalize_vietnamese(f'Khach hang: {self.client_name}'), ln=True, align='C')
        pdf.cell(0, 8, normalize_vietnamese(f'Ky bao cao: {self.period}'), ln=True, align='C')
        pdf.cell(0, 8, normalize_vietnamese(f'Ngay lap: {self.report_date}'), ln=True, align='C')
        
        pdf.ln(10)
        
        # Tổng quan metrics
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 10, normalize_vietnamese('TONG QUAN HIEU SUAT'), ln=True)
        
        pdf.set_font('Helvetica', '', 11)
        
        # Metrics table
        metrics_data = [
            ('Tong loi nhuan', f"{self.metrics.total_return:+.2f}%"),
            ('Loi nhuan hang nam', f"{self.metrics.annualized_return:+.2f}%"),
            ('Do bien dong (Volatility)', f"{self.metrics.volatility:.1f}%"),
            ('Sharpe Ratio', f"{self.metrics.sharpe_ratio:.2f}"),
            ('Max Drawdown', f"{self.metrics.max_drawdown:.1f}%"),
            ('Ty le ngay lai (Win Rate)', f"{self.metrics.win_rate:.1f}%"),
        ]
        
        col_width = 95
        row_height = 8
        
        for label, value in metrics_data:
            # Determine color based on value
            if '+' in value:
                pdf.set_text_color(0, 150, 0)  # Green
            elif '-' in value and 'Drawdown' not in label:
                pdf.set_text_color(200, 0, 0)  # Red
            else:
                pdf.set_text_color(50, 50, 50)
            
            pdf.set_font('Helvetica', '', 11)
            pdf.cell(col_width, row_height, normalize_vietnamese(label))
            pdf.set_font('Helvetica', 'B', 11)
            pdf.cell(col_width, row_height, value, ln=True)
        
        pdf.set_text_color(50, 50, 50)
        pdf.ln(10)
        
        # Performance chart
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, normalize_vietnamese('BIEU DO HIEU SUAT'), ln=True)
        
        chart_path = self._create_performance_chart()
        pdf.image(chart_path, x=10, w=190)
        
        # ============== TRANG 2: CHI TIẾT ==============
        pdf.add_page()
        
        # Chi tiết từng mã
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, normalize_vietnamese('CHI TIET DANH MUC'), ln=True)
        
        # Table header
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_fill_color(102, 126, 234)
        pdf.set_text_color(255, 255, 255)
        
        headers = ['Ma', 'Nganh', 'Ty trong', 'Loi nhuan', 'Dong gop']
        widths = [25, 45, 30, 35, 35]
        
        for header, width in zip(headers, widths):
            pdf.cell(width, 8, normalize_vietnamese(header), border=1, align='C', fill=True)
        pdf.ln()
        
        # Table data
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(50, 50, 50)
        
        for i, stock in enumerate(self.stock_metrics):
            fill = i % 2 == 0
            if fill:
                pdf.set_fill_color(245, 245, 245)
            
            pdf.cell(widths[0], 7, stock.symbol, border=1, align='C', fill=fill)
            pdf.cell(widths[1], 7, normalize_vietnamese(stock.sector[:15]), border=1, align='C', fill=fill)
            pdf.cell(widths[2], 7, f"{stock.weight}%", border=1, align='C', fill=fill)
            
            # Color for return
            ret_str = f"{stock.total_return:+.1f}%"
            pdf.cell(widths[3], 7, ret_str, border=1, align='C', fill=fill)
            
            contrib_str = f"{stock.contribution:+.1f}%"
            pdf.cell(widths[4], 7, contrib_str, border=1, align='C', fill=fill)
            pdf.ln()
        
        pdf.ln(10)
        
        # Sector allocation chart
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, normalize_vietnamese('PHAN BO THEO NGANH'), ln=True)
        
        sector_chart = self._create_sector_chart()
        pdf.image(sector_chart, x=30, w=150)
        
        # ============== TRANG 3: ĐÁNH GIÁ RỦI RO ==============
        pdf.add_page()
        
        # Monthly returns chart
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, normalize_vietnamese('LOI NHUAN THEO THANG'), ln=True)
        
        monthly_chart = self._create_monthly_chart()
        pdf.image(monthly_chart, x=10, w=190)
        
        pdf.ln(5)
        
        # Risk assessment
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, normalize_vietnamese('DANH GIA RUI RO'), ln=True)
        
        pdf.set_font('Helvetica', '', 11)
        
        for key, comment in self.risk_assessment.items():
            # Remove emoji for PDF compatibility  
            clean_comment = comment.replace('⚠️', '[!]').replace('✅', '[OK]')
            clean_comment = clean_comment.replace('📊', '').replace('🔴', '[X]')
            clean_comment = clean_comment.replace('🟡', '[!]').replace('🟢', '[OK]')
            
            pdf.multi_cell(0, 7, normalize_vietnamese(f"• {clean_comment}"))
            pdf.ln(2)
        
        pdf.ln(10)
        
        # Recommendations section
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, normalize_vietnamese('KHUYEN NGHI'), ln=True)
        
        pdf.set_font('Helvetica', '', 11)
        
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            pdf.multi_cell(0, 7, normalize_vietnamese(f"• {rec}"))
            pdf.ln(2)
        
        # ============== FOOTER ON ALL PAGES ==============
        # Go back to each page and add footer
        total_pages = pdf.page_no()
        
        for page in range(1, total_pages + 1):
            pdf.page = page
            pdf.set_y(-25)
            pdf.set_font('Helvetica', 'I', 9)
            pdf.set_text_color(128, 128, 128)
            
            footer_text = f"{self.broker_name}"
            if self.broker_phone:
                footer_text += f" | {self.broker_phone}"
            if self.broker_email:
                footer_text += f" | {self.broker_email}"
            
            pdf.cell(0, 5, normalize_vietnamese(footer_text), align='C', ln=True)
            
            if self.company_name:
                pdf.cell(0, 5, normalize_vietnamese(self.company_name), align='C', ln=True)
            
            pdf.cell(0, 5, f'Trang {page}/{total_pages}', align='C')
        
        # Save PDF
        pdf.output(output_path)
        
        # Cleanup temp files
        for temp_file in self.temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        
        return output_path
    
    def _generate_recommendations(self) -> List[str]:
        """Tạo các khuyến nghị dựa trên phân tích"""
        recommendations = []
        
        # Based on metrics
        if self.metrics.volatility > 35:
            recommendations.append(
                "Danh muc co do bien dong cao. Nen can nhac giam ty trong cac ma co beta cao "
                "hoac them cac co phieu phong thu (utilities, tieu dung thiet yeu)."
            )
        
        if self.metrics.sharpe_ratio < 0.5:
            recommendations.append(
                "Ty suat sinh loi chua tuong xung voi rui ro. Nen xem xet lai cac ma "
                "dang thua lo va chuyen sang cac co hoi tot hon."
            )
        
        if self.metrics.max_drawdown < -30:
            recommendations.append(
                "Danh muc da trai qua dot sut giam manh. Nen dat cac muc cat lo (stop-loss) "
                "va co ke hoach quan ly rui ro ro rang hon."
            )
        
        # Based on sector allocation
        max_sector = max(self.sector_alloc.values())
        if max_sector > 40:
            top_sector = max(self.sector_alloc, key=self.sector_alloc.get)
            recommendations.append(
                f"Danh muc tap trung {max_sector:.0f}% vao nganh {top_sector}. "
                f"Nen da dang hoa de giam rui ro nganh."
            )
        
        # Based on individual stocks
        losers = [s for s in self.stock_metrics if s.total_return < -20]
        if losers:
            loser_names = ', '.join([s.symbol for s in losers])
            recommendations.append(
                f"Cac ma {loser_names} dang thua lo tren 20%. "
                f"Nen danh gia lai trien vong va quyet dinh giu/ban."
            )
        
        if not recommendations:
            recommendations.append(
                "Danh muc dang hoat dong tot. Tiep tuc theo doi va tai can bang dinh ky."
            )
        
        return recommendations


# ============== COMMAND LINE INTERFACE ==============
if __name__ == "__main__":
    # Demo report generation
    print("=" * 50)
    print("DEMO: TAO BAO CAO PDF")
    print("=" * 50)
    
    # Sample portfolio
    sample_holdings = {
        'VCB': 25,
        'FPT': 20,
        'HPG': 15,
        'MWG': 15,
        'VHM': 10,
        'TCB': 15,
    }
    
    # Create report
    report = ReportGenerator(
        client_name="Nguyen Van A",
        holdings=sample_holdings,
        broker_name="Tran Van B - Moi Gioi CK",
        broker_phone="0909 123 456",
        broker_email="broker@email.com",
        company_name="Cong ty Chung khoan ABC"
    )
    
    output_file = "sample_report.pdf"
    report.generate(output_file)
    
    print(f"\n✅ Da tao bao cao: {output_file}")
    print(f"📊 Danh muc gom {len(sample_holdings)} ma")
    print(f"📈 Tong loi nhuan: {report.metrics.total_return:+.2f}%")
