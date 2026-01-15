"""
Portfolio Tool - Main Entry Point
Công cụ phân tích danh mục và tạo báo cáo cho môi giới chứng khoán

Usage:
    python main.py dashboard          # Chạy web dashboard
    python main.py report             # Tạo báo cáo PDF
    python main.py analyze            # Phân tích nhanh trên terminal
"""

import argparse
import sys
import os
from datetime import datetime, timedelta


def run_dashboard():
    """Khởi động Streamlit Dashboard"""
    print("🚀 Đang khởi động Dashboard...")
    print("📝 Mở trình duyệt tại: http://localhost:8501")
    os.system("streamlit run dashboard.py")


def run_report():
    """Tạo báo cáo PDF qua interactive prompt"""
    from report_generator import ReportGenerator
    
    print("=" * 50)
    print("📄 TẠO BÁO CÁO DANH MỤC ĐẦU TƯ")
    print("=" * 50)
    
    # Input client info
    client_name = input("\n👤 Tên khách hàng: ").strip() or "Khach Hang"
    
    # Input holdings
    print("\n📊 Nhập danh mục (nhấn Enter 2 lần để kết thúc):")
    print("   Format: MÃ TỶ_TRỌNG (VD: VCB 30)")
    
    holdings = {}
    while True:
        line = input("   > ").strip()
        if not line:
            if holdings:
                break
            print("   ⚠️ Cần ít nhất 1 mã!")
            continue
        
        try:
            parts = line.split()
            symbol = parts[0].upper()
            weight = float(parts[1]) if len(parts) > 1 else 10
            holdings[symbol] = weight
            print(f"   ✓ Đã thêm {symbol}: {weight}%")
        except Exception as e:
            print(f"   ❌ Lỗi format. Thử lại!")
    
    # Broker info
    print("\n📋 Thông tin môi giới (Enter để bỏ qua):")
    broker_name = input("   Tên: ").strip() or ""
    broker_phone = input("   SĐT: ").strip() or ""
    broker_email = input("   Email: ").strip() or ""
    company = input("   Công ty: ").strip() or ""
    
    # Generate report
    print("\n⏳ Đang tạo báo cáo...")
    
    # Create output filename
    safe_name = "".join(c for c in client_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_')
    output_file = f"report_{safe_name}_{datetime.now().strftime('%Y%m%d')}.pdf"
    
    try:
        report = ReportGenerator(
            client_name=client_name,
            holdings=holdings,
            broker_name=broker_name,
            broker_phone=broker_phone,
            broker_email=broker_email,
            company_name=company
        )
        
        report.generate(output_file)
        
        print(f"\n✅ Đã tạo báo cáo: {output_file}")
        print(f"📈 Tổng lợi nhuận: {report.metrics.total_return:+.2f}%")
        print(f"📊 Sharpe Ratio: {report.metrics.sharpe_ratio:.2f}")
        
    except Exception as e:
        print(f"\n❌ Lỗi tạo báo cáo: {e}")
        sys.exit(1)


def run_analyze():
    """Phân tích nhanh danh mục trên terminal"""
    from analysis_engine import PortfolioAnalyzer
    
    print("=" * 50)
    print("📊 PHÂN TÍCH NHANH DANH MỤC")
    print("=" * 50)
    
    # Input holdings
    print("\nNhập danh mục (format: MÃ TỶ_TRỌNG, Enter 2 lần để kết thúc):")
    
    holdings = {}
    while True:
        line = input("> ").strip()
        if not line:
            if holdings:
                break
            continue
        
        try:
            parts = line.split()
            symbol = parts[0].upper()
            weight = float(parts[1]) if len(parts) > 1 else 10
            holdings[symbol] = weight
        except:
            print("❌ Lỗi format!")
    
    print("\n⏳ Đang phân tích...")
    
    try:
        analyzer = PortfolioAnalyzer(holdings)
        metrics = analyzer.get_portfolio_metrics()
        stocks = analyzer.get_stock_metrics()
        sectors = analyzer.get_sector_allocation()
        risks = analyzer.get_risk_assessment()
        
        print("\n" + "=" * 50)
        print("📈 KẾT QUẢ PHÂN TÍCH")
        print("=" * 50)
        
        print(f"\n🎯 TỔNG QUAN:")
        print(f"   Tổng lợi nhuận:      {metrics.total_return:+.2f}%")
        print(f"   Lợi nhuận/năm:       {metrics.annualized_return:+.2f}%")
        print(f"   Độ biến động:        {metrics.volatility:.1f}%")
        print(f"   Sharpe Ratio:        {metrics.sharpe_ratio:.2f}")
        print(f"   Max Drawdown:        {metrics.max_drawdown:.1f}%")
        
        print(f"\n📊 CHI TIẾT TỪNG MÃ:")
        for s in stocks:
            color = "🟢" if s.total_return >= 0 else "🔴"
            print(f"   {color} {s.symbol:6} | {s.sector:12} | {s.weight:5.1f}% | {s.total_return:+7.1f}%")
        
        print(f"\n🏢 PHÂN BỔ NGÀNH:")
        for sector, weight in sectors.items():
            print(f"   {sector:15} {weight:5.1f}%")
        
        print(f"\n⚠️ ĐÁNH GIÁ RỦI RO:")
        for comment in risks.values():
            print(f"   {comment}")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        sys.exit(1)


def run_batch_reports():
    """Tạo báo cáo hàng loạt từ file CSV"""
    import pandas as pd
    from report_generator import ReportGenerator
    
    print("=" * 50)
    print("📄 TẠO BÁO CÁO HÀNG LOẠT")
    print("=" * 50)
    
    print("""
    File CSV cần có format:
    client_name,symbol1,weight1,symbol2,weight2,...
    
    Ví dụ:
    Nguyen Van A,VCB,30,FPT,25,HPG,20
    Tran Van B,TCB,40,MWG,30,VHM,30
    """)
    
    file_path = input("Đường dẫn file CSV: ").strip()
    
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file: {file_path}")
        return
    
    # Broker info (applied to all reports)
    print("\n📋 Thông tin môi giới (áp dụng cho tất cả báo cáo):")
    broker_name = input("   Tên: ").strip() or ""
    broker_phone = input("   SĐT: ").strip() or ""
    
    # Read CSV and generate reports
    try:
        df = pd.read_csv(file_path, header=None)
        
        output_dir = "reports_" + datetime.now().strftime('%Y%m%d_%H%M')
        os.makedirs(output_dir, exist_ok=True)
        
        success = 0
        failed = 0
        
        for idx, row in df.iterrows():
            client_name = row[0]
            
            # Parse holdings from remaining columns
            holdings = {}
            for i in range(1, len(row), 2):
                if pd.notna(row[i]) and i+1 < len(row) and pd.notna(row[i+1]):
                    holdings[str(row[i]).upper()] = float(row[i+1])
            
            if not holdings:
                print(f"⚠️ Bỏ qua {client_name}: không có danh mục")
                continue
            
            try:
                print(f"📝 Đang tạo báo cáo cho {client_name}...")
                
                safe_name = "".join(c for c in client_name if c.isalnum() or c in (' ', '-', '_')).strip()
                output_file = os.path.join(output_dir, f"{safe_name.replace(' ', '_')}.pdf")
                
                report = ReportGenerator(
                    client_name=client_name,
                    holdings=holdings,
                    broker_name=broker_name,
                    broker_phone=broker_phone
                )
                report.generate(output_file)
                success += 1
                
            except Exception as e:
                print(f"❌ Lỗi với {client_name}: {e}")
                failed += 1
        
        print(f"\n✅ Hoàn thành: {success} báo cáo")
        if failed:
            print(f"❌ Thất bại: {failed}")
        print(f"📁 Thư mục output: {output_dir}")
        
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio Tool - Công cụ phân tích danh mục cho môi giới chứng khoán"
    )
    
    parser.add_argument(
        'command',
        choices=['dashboard', 'report', 'analyze', 'batch'],
        nargs='?',
        default='dashboard',
        help="""
        dashboard: Chạy web dashboard (mặc định)
        report: Tạo báo cáo PDF cho 1 khách
        analyze: Phân tích nhanh trên terminal
        batch: Tạo báo cáo hàng loạt từ CSV
        """
    )
    
    args = parser.parse_args()
    
    if args.command == 'dashboard':
        run_dashboard()
    elif args.command == 'report':
        run_report()
    elif args.command == 'analyze':
        run_analyze()
    elif args.command == 'batch':
        run_batch_reports()


if __name__ == "__main__":
    main()
