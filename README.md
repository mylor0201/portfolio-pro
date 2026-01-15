# 📊 Portfolio Tool for Stock Brokers

Công cụ phân tích danh mục đầu tư và tạo báo cáo chuyên nghiệp cho môi giới chứng khoán Việt Nam.

## ✨ Tính năng

### 🖥️ Project 1: Portfolio Health Check Dashboard
- Nhập danh mục thủ công hoặc upload CSV
- Tính toán các chỉ số: Return, Volatility, Sharpe Ratio, Max Drawdown
- So sánh hiệu suất với VN-Index
- Biểu đồ tương tác (Plotly)
- Phân tích phân bổ ngành
- Đánh giá rủi ro tự động

### 📄 Project 3: Client Report Generator
- Tạo báo cáo PDF chuyên nghiệp
- Tự động generate biểu đồ
- Thông tin broker customizable
- Khuyến nghị tự động dựa trên phân tích
- Hỗ trợ tạo báo cáo hàng loạt

---

## 🚀 Cài đặt

### Bước 1: Clone/Download project

```bash
# Nếu dùng git
git clone <your-repo>
cd portfolio_tool

# Hoặc download và giải nén
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

---

## 📖 Cách sử dụng

### 1. Chạy Web Dashboard

```bash
python main.py dashboard
# hoặc
streamlit run dashboard.py
```

Mở trình duyệt tại `http://localhost:8501`

**Hướng dẫn:**
1. Nhập số lượng mã cổ phiếu ở sidebar trái
2. Điền mã và tỷ trọng từng mã
3. Chọn khoảng thời gian phân tích
4. Xem kết quả phân tích
5. Export CSV hoặc tạo PDF

### 2. Tạo báo cáo PDF (CLI)

```bash
python main.py report
```

Làm theo hướng dẫn trên màn hình:
- Nhập tên khách hàng
- Nhập danh mục (format: MÃ TỶ_TRỌNG)
- Nhập thông tin môi giới
- File PDF sẽ được tạo trong thư mục hiện tại

### 3. Phân tích nhanh (Terminal)

```bash
python main.py analyze
```

### 4. Tạo báo cáo hàng loạt

```bash
python main.py batch
```

Chuẩn bị file CSV với format:
```csv
Nguyen Van A,VCB,30,FPT,25,HPG,20,MWG,15,VHM,10
Tran Van B,TCB,40,MBB,30,VNM,30
```

---

## 📁 Cấu trúc project

```
portfolio_tool/
├── main.py              # Entry point chính
├── dashboard.py         # Streamlit web app
├── report_generator.py  # Tạo PDF báo cáo
├── analysis_engine.py   # Logic phân tích
├── data_layer.py        # Lấy dữ liệu chứng khoán
├── requirements.txt     # Dependencies
└── README.md           # File này
```

---

## 🎨 Customize cho công việc của bạn

### Thêm logo công ty vào báo cáo

Mở `report_generator.py`, tìm method `generate()` và thêm:

```python
# Sau dòng pdf.add_page()
pdf.image('path/to/your/logo.png', x=10, y=10, w=30)
```

### Thay đổi màu sắc Dashboard

Mở `dashboard.py`, chỉnh sửa phần CSS:

```python
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #YOUR_COLOR1 0%, #YOUR_COLOR2 100%);
        ...
    }
</style>
""", unsafe_allow_html=True)
```

### Thêm ngành mới vào SECTOR_MAPPING

Mở `data_layer.py`, thêm vào dictionary `SECTOR_MAPPING`:

```python
SECTOR_MAPPING = {
    ...
    'NEW_STOCK': 'Ngành Mới',
}
```

---

## 💡 Use Cases cho Môi Giới

### 1. Gặp khách hàng mới
- Yêu cầu khách chia sẻ danh mục hiện tại
- Nhập vào Dashboard
- Show kết quả phân tích ngay trên laptop/tablet
- "Anh/chị thấy đó, danh mục đang tập trung quá nhiều vào ngân hàng..."

### 2. Chăm sóc khách hàng định kỳ
- Chạy `python main.py batch` với danh sách khách
- Gửi PDF qua email/Zalo hàng tháng
- "Em gửi anh/chị báo cáo tháng này ạ"

### 3. Tư vấn tái cân bằng
- So sánh nhiều kịch bản danh mục
- Show sự khác biệt về Sharpe Ratio, Volatility
- Đề xuất điều chỉnh dựa trên data

### 4. Content Marketing
- Screenshot biểu đồ từ Dashboard
- Post lên Facebook/Zalo với nhận định
- Thu hút khách hàng tiềm năng

---

## 🔧 Troubleshooting

### Lỗi "Module not found"
```bash
pip install -r requirements.txt --force-reinstall
```

### Lỗi lấy dữ liệu chứng khoán
- Kiểm tra kết nối internet
- vnstock có thể đang maintenance
- Tool sẽ tự dùng mock data để demo

### Lỗi font tiếng Việt trong PDF
- PDF hiện dùng font ASCII
- Để hỗ trợ tiếng Việt đầy đủ, cần thêm font DejaVu:
```python
# Trong report_generator.py
pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
pdf.set_font('DejaVu', '', 12)
```

---

## 📈 Roadmap phát triển

- [ ] Tích hợp Telegram bot gửi alert
- [ ] Thêm technical indicators (RSI, MACD)
- [ ] Multi-language PDF (Vietnamese với dấu)
- [ ] Database lưu lịch sử khách hàng
- [ ] Authentication cho dashboard
- [ ] Mobile-responsive design

---

## 📞 Liên hệ

Được phát triển bởi: [Tên của bạn]
Email: [email]
Phone: [số điện thoại]

---

## 📜 License

MIT License - Tự do sử dụng và chỉnh sửa cho mục đích cá nhân và thương mại.
