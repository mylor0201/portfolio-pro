"""
Data Layer - Lấy dữ liệu chứng khoán Việt Nam
Sử dụng vnstock library hoặc fallback sang mock data để test
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict

# Mapping sector cho các mã phổ biến (mở rộng theo nhu cầu)
SECTOR_MAPPING = {
    # Ngân hàng
    'VCB': 'Ngân hàng', 'BID': 'Ngân hàng', 'CTG': 'Ngân hàng',
    'TCB': 'Ngân hàng', 'MBB': 'Ngân hàng', 'VPB': 'Ngân hàng',
    'ACB': 'Ngân hàng', 'HDB': 'Ngân hàng', 'STB': 'Ngân hàng',
    
    # Bất động sản
    'VHM': 'Bất động sản', 'VIC': 'Bất động sản', 'NVL': 'Bất động sản',
    'PDR': 'Bất động sản', 'DXG': 'Bất động sản', 'KDH': 'Bất động sản',
    
    # Chứng khoán
    'SSI': 'Chứng khoán', 'VND': 'Chứng khoán', 'HCM': 'Chứng khoán',
    'VCI': 'Chứng khoán', 'SHS': 'Chứng khoán',
    
    # Thép
    'HPG': 'Thép', 'HSG': 'Thép', 'NKG': 'Thép',
    
    # Công nghệ
    'FPT': 'Công nghệ', 'CMG': 'Công nghệ',
    
    # Bán lẻ
    'MWG': 'Bán lẻ', 'FRT': 'Bán lẻ', 'PNJ': 'Bán lẻ',
    
    # Dầu khí
    'GAS': 'Dầu khí', 'PLX': 'Dầu khí', 'PVD': 'Dầu khí',
    'PVS': 'Dầu khí', 'BSR': 'Dầu khí',
    
    # Điện
    'POW': 'Điện', 'GEG': 'Điện', 'REE': 'Điện',
    
    # Thực phẩm
    'VNM': 'Thực phẩm', 'MSN': 'Thực phẩm', 'SAB': 'Thực phẩm',
    
    # Hàng không
    'VJC': 'Hàng không', 'HVN': 'Hàng không',
}


def get_stock_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Lấy dữ liệu giá lịch sử của một mã cổ phiếu
    
    Parameters:
    - symbol: Mã cổ phiếu (VD: 'VCB', 'FPT')
    - start_date: Ngày bắt đầu format 'YYYY-MM-DD'
    - end_date: Ngày kết thúc format 'YYYY-MM-DD'
    
    Returns:
    - DataFrame với columns: date, open, high, low, close, volume
    """
    try:
        from vnstock import Quote
        
        # Sử dụng VCI source (Viet Capital Securities)
        quote = Quote(symbol=symbol, source='VCI')
        df = quote.history(start=start_date, end=end_date, interval='1D')
        
        if df is None or len(df) == 0:
            raise ValueError(f"Không có dữ liệu cho {symbol}")
        
        # Chuẩn hóa column names (vnstock 3.0+ có thể dùng tên khác)
        column_mapping = {
            'time': 'date',
            'TradingDate': 'date',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'open': 'open',
            'high': 'high',
            'low': 'low', 
            'close': 'close',
            'volume': 'volume'
        }
        
        # Rename columns that exist
        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu {symbol}: {e}")
        print("Sử dụng mock data để demo...")
        return generate_mock_data(symbol, start_date, end_date)


def generate_mock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Tạo dữ liệu giả để test khi không có kết nối API
    """
    np.random.seed(hash(symbol) % 2**32)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    n = len(dates)
    
    # Random walk for price
    base_price = np.random.uniform(20, 100)
    returns = np.random.normal(0.0005, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices * np.random.uniform(0.98, 1.0, n),
        'high': prices * np.random.uniform(1.0, 1.03, n),
        'low': prices * np.random.uniform(0.97, 1.0, n),
        'close': prices,
        'volume': np.random.randint(100000, 5000000, n)
    })
    
    return df


def get_vnindex_data(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Lấy dữ liệu VN-Index để làm benchmark
    """
    try:
        from vnstock import Quote
        
        # VN-Index sử dụng VCI source
        quote = Quote(symbol='VNINDEX', source='VCI')
        df = quote.history(start=start_date, end=end_date, interval='1D')
        
        if df is None or len(df) == 0:
            raise ValueError("Không có dữ liệu VNINDEX")
        
        # Chuẩn hóa column names
        column_mapping = {
            'time': 'date',
            'TradingDate': 'date',
            'Close': 'close',
            'close': 'close'
        }
        
        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        return df[['date', 'close']]
        
    except Exception as e:
        print(f"Lỗi khi lấy VN-Index: {e}")
        # Mock VN-Index data
        return generate_mock_data('VNINDEX', start_date, end_date)[['date', 'close']]


def get_sector(symbol: str) -> str:
    """
    Lấy ngành của một mã cổ phiếu
    """
    return SECTOR_MAPPING.get(symbol.upper(), 'Khác')


def get_multiple_stocks(symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Lấy dữ liệu nhiều mã cùng lúc
    
    Returns:
    - Dictionary với key là symbol, value là DataFrame
    """
    result = {}
    for symbol in symbols:
        df = get_stock_data(symbol, start_date, end_date)
        if df is not None and len(df) > 0:
            result[symbol.upper()] = df
    return result


# ============== TEST ==============
if __name__ == "__main__":
    # Test lấy dữ liệu
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print("Testing data fetch...")
    df = get_stock_data('VCB', start, end)
    print(f"VCB data shape: {df.shape}")
    print(df.head())
    
    print(f"\nSector of VCB: {get_sector('VCB')}")
    print(f"Sector of FPT: {get_sector('FPT')}")
