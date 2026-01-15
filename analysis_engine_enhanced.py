"""
Enhanced Portfolio Analysis Engine
Handles portfolios with different entry dates and proper performance attribution

Key Features:
- Time-Weighted Return (TWR) calculation
- Money-Weighted Return (MWR/IRR) calculation  
- Proper benchmark alignment for different entry dates
- Portfolio attribution analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy import optimize
import warnings

warnings.filterwarnings('ignore')


@dataclass
class StockHolding:
    """Represents a single stock holding with entry information"""
    symbol: str
    weight: float  # Target weight in portfolio (%)
    entry_date: Optional[datetime] = None
    entry_price: Optional[float] = None
    quantity: Optional[int] = None
    current_price: Optional[float] = None
    sector: str = "Khác"
    
    @property
    def holding_value(self) -> float:
        """Calculate current holding value"""
        if self.quantity and self.current_price:
            return self.quantity * self.current_price
        return 0.0
    
    @property
    def cost_basis(self) -> float:
        """Calculate cost basis"""
        if self.quantity and self.entry_price:
            return self.quantity * self.entry_price
        return 0.0
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        return self.holding_value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage"""
        if self.cost_basis > 0:
            return (self.unrealized_pnl / self.cost_basis) * 100
        return 0.0


@dataclass
class PerformanceMetrics:
    """Comprehensive portfolio performance metrics"""
    # Return metrics
    total_return: float = 0.0
    twr: float = 0.0  # Time-Weighted Return
    mwr: float = 0.0  # Money-Weighted Return (IRR)
    annualized_return: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional VaR 95%
    
    # Performance statistics
    win_rate: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    avg_daily_return: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # Benchmark comparison
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    r_squared: float = 0.0
    
    # Additional metrics
    up_capture: float = 0.0  # Upside capture ratio
    down_capture: float = 0.0  # Downside capture ratio


@dataclass
class SubPeriod:
    """Represents a sub-period for TWR calculation"""
    start_date: datetime
    end_date: datetime
    start_value: float
    end_value: float
    cash_flow: float = 0.0
    holdings: Dict[str, float] = field(default_factory=dict)
    
    @property
    def return_pct(self) -> float:
        """Calculate return for this sub-period"""
        adjusted_start = self.start_value + self.cash_flow
        if adjusted_start > 0:
            return (self.end_value / adjusted_start) - 1
        return 0.0


class EnhancedPortfolioAnalyzer:
    """
    Enhanced Portfolio Analyzer with support for:
    - Different entry dates per stock
    - Time-Weighted Return (TWR)
    - Money-Weighted Return (MWR/IRR)
    - Proper benchmark alignment
    - Attribution analysis
    """
    
    def __init__(
        self,
        holdings: List[StockHolding],
        benchmark_symbol: str = "VNINDEX",
        risk_free_rate: float = 0.05,  # Annual risk-free rate
        analysis_start: Optional[datetime] = None,
        analysis_end: Optional[datetime] = None
    ):
        self.holdings = holdings
        self.benchmark_symbol = benchmark_symbol
        self.risk_free_rate = risk_free_rate
        
        # Determine analysis period
        entry_dates = [h.entry_date for h in holdings if h.entry_date]
        if entry_dates:
            self.first_entry = min(entry_dates)
        else:
            self.first_entry = analysis_start or datetime.now() - timedelta(days=365)
        
        self.analysis_start = analysis_start or self.first_entry
        self.analysis_end = analysis_end or datetime.now()
        
        # Data storage
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None
        self.portfolio_values: Optional[pd.DataFrame] = None
        
        # Sub-periods for TWR calculation
        self.sub_periods: List[SubPeriod] = []
        
        # Cash flows for MWR calculation
        self.cash_flows: List[Tuple[datetime, float]] = []
        
    def _fetch_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch stock price data
        Replace this with actual data fetching (e.g., from vnstock)
        """
        # Mock data generation for demonstration
        np.random.seed(hash(symbol) % 2**32)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n_days = len(dates)
        
        # Generate realistic price series
        base_price = np.random.uniform(20, 150) * 1000
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = base_price * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.015, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.015, n_days))),
            'volume': np.random.randint(100000, 10000000, n_days)
        })
        
        return df.set_index('date')
    
    def _fetch_benchmark_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch benchmark index data"""
        np.random.seed(42)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n_days = len(dates)
        
        # Generate benchmark with lower volatility
        base_value = 1200
        returns = np.random.normal(0.0003, 0.012, n_days)
        values = base_value * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'date': dates,
            'close': values
        })
        
        return df.set_index('date')
    
    def load_data(self):
        """Load all required price data"""
        # Load benchmark data from first entry date
        self.benchmark_data = self._fetch_benchmark_data(
            self.first_entry, 
            self.analysis_end
        )
        
        # Load stock data for each holding
        for holding in self.holdings:
            start = holding.entry_date or self.analysis_start
            self.stock_data[holding.symbol] = self._fetch_stock_data(
                holding.symbol,
                start,
                self.analysis_end
            )
            
            # Update current price
            if holding.symbol in self.stock_data:
                holding.current_price = self.stock_data[holding.symbol]['close'].iloc[-1]
    
    def _identify_sub_periods(self):
        """
        Identify sub-periods based on cash flows (new entries)
        Each new stock entry creates a new sub-period
        """
        # Get all unique entry dates
        entry_dates = sorted(set(
            h.entry_date for h in self.holdings 
            if h.entry_date and h.entry_date >= self.analysis_start
        ))
        
        if not entry_dates:
            entry_dates = [self.analysis_start]
        
        # Add analysis end date
        all_dates = entry_dates + [self.analysis_end]
        
        # Create sub-periods
        self.sub_periods = []
        
        for i in range(len(all_dates) - 1):
            start = all_dates[i]
            end = all_dates[i + 1]
            
            # Calculate holdings active in this period
            active_holdings = {
                h.symbol: h.weight 
                for h in self.holdings 
                if h.entry_date and h.entry_date <= start
            }
            
            # Calculate cash flow (new entries at start of period)
            cash_flow = sum(
                h.cost_basis 
                for h in self.holdings 
                if h.entry_date == start
            )
            
            sub_period = SubPeriod(
                start_date=start,
                end_date=end,
                start_value=0,  # Will be calculated
                end_value=0,    # Will be calculated
                cash_flow=cash_flow,
                holdings=active_holdings
            )
            
            self.sub_periods.append(sub_period)
            
            # Record cash flow for MWR calculation
            if cash_flow > 0:
                self.cash_flows.append((start, cash_flow))
    
    def calculate_twr(self) -> float:
        """
        Calculate Time-Weighted Return using chain-linking
        
        TWR = [(1 + R1) × (1 + R2) × ... × (1 + Rn)] - 1
        
        Where Ri is the return for sub-period i
        """
        if not self.sub_periods:
            self._identify_sub_periods()
        
        # Calculate return for each sub-period
        twr = 1.0
        
        for sub_period in self.sub_periods:
            # Get portfolio value at start and end of sub-period
            start_value = self._calculate_portfolio_value(
                sub_period.start_date, 
                sub_period.holdings
            )
            end_value = self._calculate_portfolio_value(
                sub_period.end_date, 
                sub_period.holdings
            )
            
            # Adjust for cash flow
            adjusted_start = start_value + sub_period.cash_flow
            
            if adjusted_start > 0:
                period_return = end_value / adjusted_start
                twr *= period_return
        
        return (twr - 1) * 100
    
    def calculate_mwr(self) -> float:
        """
        Calculate Money-Weighted Return (Internal Rate of Return)
        
        Uses Newton-Raphson method to solve:
        Σ CFi × (1 + r)^(T - ti) = Final Value
        """
        if not self.cash_flows:
            self._identify_sub_periods()
        
        if not self.cash_flows:
            return 0.0
        
        # Get final portfolio value
        final_value = self._calculate_portfolio_value(
            self.analysis_end,
            {h.symbol: h.weight for h in self.holdings}
        )
        
        # Calculate total days
        first_cf_date = self.cash_flows[0][0]
        total_days = (self.analysis_end - first_cf_date).days
        
        if total_days <= 0:
            return 0.0
        
        def npv(r):
            """Calculate NPV for given rate r"""
            total = 0
            for date, cf in self.cash_flows:
                days_to_end = (self.analysis_end - date).days
                years = days_to_end / 365
                total += cf * ((1 + r) ** years)
            return total - final_value
        
        try:
            # Solve for IRR using Newton-Raphson
            irr = optimize.newton(npv, 0.1, maxiter=100)
            return irr * 100
        except:
            # Fallback to Modified Dietz
            return self._calculate_modified_dietz()
    
    def _calculate_modified_dietz(self) -> float:
        """
        Calculate Modified Dietz Return as fallback for MWR
        
        MDR = (End Value - Start Value - Cash Flows) / (Start Value + Weighted Cash Flows)
        """
        if not self.cash_flows:
            return 0.0
        
        total_cf = sum(cf for _, cf in self.cash_flows)
        final_value = self._calculate_portfolio_value(
            self.analysis_end,
            {h.symbol: h.weight for h in self.holdings}
        )
        
        first_date = self.cash_flows[0][0]
        total_days = (self.analysis_end - first_date).days
        
        if total_days <= 0 or total_cf == 0:
            return 0.0
        
        # Calculate weighted cash flows
        weighted_cf = 0
        for date, cf in self.cash_flows:
            days_held = (self.analysis_end - date).days
            weight = days_held / total_days
            weighted_cf += cf * weight
        
        # Modified Dietz return
        denominator = weighted_cf
        if denominator == 0:
            return 0.0
        
        return ((final_value - total_cf) / denominator) * 100
    
    def _calculate_portfolio_value(
        self, 
        date: datetime, 
        holdings: Dict[str, float]
    ) -> float:
        """Calculate portfolio value at a specific date"""
        total_value = 0
        
        for symbol, weight in holdings.items():
            if symbol in self.stock_data:
                df = self.stock_data[symbol]
                
                # Find closest available date
                available_dates = df.index[df.index <= date]
                if len(available_dates) > 0:
                    closest_date = available_dates[-1]
                    price = df.loc[closest_date, 'close']
                    
                    # Find corresponding holding
                    holding = next(
                        (h for h in self.holdings if h.symbol == symbol), 
                        None
                    )
                    
                    if holding and holding.quantity:
                        total_value += holding.quantity * price
                    else:
                        # Use weight-based calculation
                        total_value += weight * 1000000 / 100  # Assume 1M portfolio
        
        return total_value
    
    def calculate_benchmark_aligned_return(self) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate benchmark return aligned with portfolio entry dates
        
        For fair comparison, we align VN-Index return starting from
        the earliest entry date, weighted by capital allocation
        """
        if self.benchmark_data is None:
            return pd.Series(), pd.Series()
        
        # Get benchmark prices
        benchmark_prices = self.benchmark_data['close']
        
        # Calculate simple cumulative return from first entry
        first_price = benchmark_prices.iloc[0]
        benchmark_returns = (benchmark_prices / first_price - 1) * 100
        
        # Calculate portfolio cumulative returns
        portfolio_returns = self._calculate_portfolio_returns()
        
        return portfolio_returns, benchmark_returns
    
    def _calculate_portfolio_returns(self) -> pd.Series:
        """Calculate portfolio cumulative returns"""
        if not self.stock_data:
            return pd.Series()
        
        # Get all available dates
        all_dates = set()
        for df in self.stock_data.values():
            all_dates.update(df.index)
        
        dates = sorted(all_dates)
        
        # Calculate portfolio value for each date
        portfolio_values = []
        
        for date in dates:
            # Determine active holdings at this date
            active_holdings = {
                h.symbol: h.weight
                for h in self.holdings
                if h.entry_date is None or h.entry_date <= date
            }
            
            if active_holdings:
                value = self._calculate_portfolio_value(date, active_holdings)
                portfolio_values.append((date, value))
        
        if not portfolio_values:
            return pd.Series()
        
        df = pd.DataFrame(portfolio_values, columns=['date', 'value'])
        df = df.set_index('date')
        
        # Calculate cumulative returns
        first_value = df['value'].iloc[0]
        returns = (df['value'] / first_value - 1) * 100
        
        return returns
    
    def calculate_contribution_analysis(self) -> pd.DataFrame:
        """
        Calculate return contribution of each stock
        Weighted by holding period and capital allocation
        """
        contributions = []
        
        total_portfolio_return = self.calculate_twr()
        
        for holding in self.holdings:
            if holding.symbol in self.stock_data:
                df = self.stock_data[holding.symbol]
                
                # Get entry price
                if holding.entry_date and holding.entry_date in df.index:
                    entry_price = df.loc[holding.entry_date, 'close']
                else:
                    entry_price = holding.entry_price or df['close'].iloc[0]
                
                current_price = df['close'].iloc[-1]
                
                # Calculate stock return
                stock_return = (current_price / entry_price - 1) * 100
                
                # Calculate contribution (weight × return)
                contribution = (holding.weight / 100) * stock_return
                
                # Calculate holding period
                if holding.entry_date:
                    holding_days = (self.analysis_end - holding.entry_date).days
                else:
                    holding_days = (self.analysis_end - self.analysis_start).days
                
                contributions.append({
                    'symbol': holding.symbol,
                    'sector': holding.sector,
                    'weight': holding.weight,
                    'entry_date': holding.entry_date,
                    'holding_days': holding_days,
                    'stock_return': stock_return,
                    'contribution': contribution,
                    'contribution_pct': (contribution / total_portfolio_return * 100) 
                        if total_portfolio_return != 0 else 0
                })
        
        return pd.DataFrame(contributions)
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        portfolio_returns = self._calculate_portfolio_returns()
        
        if portfolio_returns.empty:
            return {}
        
        daily_returns = portfolio_returns.pct_change().dropna()
        
        if daily_returns.empty:
            return {}
        
        # Basic statistics
        mean_return = daily_returns.mean()
        std_return = daily_returns.std()
        
        # Annualized metrics
        ann_return = mean_return * 252 * 100
        ann_volatility = std_return * np.sqrt(252) * 100
        
        # Sharpe Ratio
        daily_rf = self.risk_free_rate / 252
        sharpe = (mean_return - daily_rf) / std_return * np.sqrt(252) if std_return > 0 else 0
        
        # Sortino Ratio (using downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std()
        sortino = (mean_return - daily_rf) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Max Drawdown
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        # VaR and CVaR
        var_95 = np.percentile(daily_returns, 5) * 100
        cvar_95 = daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean() * 100
        
        # Win rate
        win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100
        
        return {
            'annualized_return': ann_return,
            'annualized_volatility': ann_volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'best_day': daily_returns.max() * 100,
            'worst_day': daily_returns.min() * 100,
            'skewness': daily_returns.skew(),
            'kurtosis': daily_returns.kurtosis()
        }
    
    def calculate_benchmark_comparison(self) -> Dict[str, float]:
        """Calculate alpha, beta, and other benchmark comparison metrics"""
        portfolio_returns, benchmark_returns = self.calculate_benchmark_aligned_return()
        
        if portfolio_returns.empty or benchmark_returns.empty:
            return {}
        
        # Align dates
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        port_ret = portfolio_returns.loc[common_dates].pct_change().dropna()
        bench_ret = benchmark_returns.loc[common_dates].pct_change().dropna()
        
        if port_ret.empty or bench_ret.empty:
            return {}
        
        # Ensure same length
        min_len = min(len(port_ret), len(bench_ret))
        port_ret = port_ret.iloc[:min_len]
        bench_ret = bench_ret.iloc[:min_len]
        
        # Calculate beta
        covariance = np.cov(port_ret, bench_ret)[0, 1]
        benchmark_variance = np.var(bench_ret)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
        
        # Calculate alpha (Jensen's alpha)
        port_ann = port_ret.mean() * 252
        bench_ann = bench_ret.mean() * 252
        alpha = (port_ann - (self.risk_free_rate + beta * (bench_ann - self.risk_free_rate))) * 100
        
        # Tracking error
        tracking_diff = port_ret - bench_ret
        tracking_error = tracking_diff.std() * np.sqrt(252) * 100
        
        # Information ratio
        ir = (port_ann - bench_ann) / (tracking_error / 100) if tracking_error > 0 else 0
        
        # R-squared
        correlation = np.corrcoef(port_ret, bench_ret)[0, 1]
        r_squared = correlation ** 2
        
        # Capture ratios
        up_periods = bench_ret > 0
        down_periods = bench_ret < 0
        
        up_capture = (port_ret[up_periods].mean() / bench_ret[up_periods].mean() * 100) \
            if up_periods.any() and bench_ret[up_periods].mean() != 0 else 0
        down_capture = (port_ret[down_periods].mean() / bench_ret[down_periods].mean() * 100) \
            if down_periods.any() and bench_ret[down_periods].mean() != 0 else 0
        
        return {
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': ir,
            'r_squared': r_squared,
            'up_capture': up_capture,
            'down_capture': down_capture
        }
    
    def get_full_analysis(self) -> Dict[str, Any]:
        """Get comprehensive portfolio analysis"""
        # Load data if not already loaded
        if not self.stock_data:
            self.load_data()
        
        # Calculate all metrics
        twr = self.calculate_twr()
        mwr = self.calculate_mwr()
        risk_metrics = self.calculate_risk_metrics()
        benchmark_metrics = self.calculate_benchmark_comparison()
        contributions = self.calculate_contribution_analysis()
        
        # Portfolio returns for charts
        portfolio_returns, benchmark_returns = self.calculate_benchmark_aligned_return()
        
        # Compile results
        return {
            'performance': {
                'twr': twr,
                'mwr': mwr,
                'total_return': twr,  # Use TWR as primary return metric
                **risk_metrics
            },
            'benchmark_comparison': benchmark_metrics,
            'contributions': contributions.to_dict('records'),
            'time_series': {
                'portfolio_returns': portfolio_returns,
                'benchmark_returns': benchmark_returns
            },
            'holdings': [
                {
                    'symbol': h.symbol,
                    'weight': h.weight,
                    'entry_date': h.entry_date,
                    'entry_price': h.entry_price,
                    'current_price': h.current_price,
                    'sector': h.sector,
                    'unrealized_pnl': h.unrealized_pnl,
                    'unrealized_pnl_pct': h.unrealized_pnl_pct
                }
                for h in self.holdings
            ]
        }


# ============== USAGE EXAMPLE ==============
if __name__ == "__main__":
    # Create sample holdings with different entry dates
    holdings = [
        StockHolding(
            symbol="VCB",
            weight=30,
            entry_date=datetime(2024, 1, 15),
            entry_price=85500,
            quantity=1000,
            sector="Ngân hàng"
        ),
        StockHolding(
            symbol="FPT",
            weight=25,
            entry_date=datetime(2024, 2, 1),
            entry_price=120000,
            quantity=500,
            sector="Công nghệ"
        ),
        StockHolding(
            symbol="HPG",
            weight=20,
            entry_date=datetime(2024, 3, 10),
            entry_price=28500,
            quantity=2000,
            sector="Thép"
        ),
        StockHolding(
            symbol="MWG",
            weight=15,
            entry_date=datetime(2024, 1, 20),
            entry_price=55000,
            quantity=800,
            sector="Bán lẻ"
        ),
        StockHolding(
            symbol="VHM",
            weight=10,
            entry_date=datetime(2024, 2, 15),
            entry_price=42000,
            quantity=1200,
            sector="Bất động sản"
        ),
    ]
    
    # Create analyzer
    analyzer = EnhancedPortfolioAnalyzer(
        holdings=holdings,
        benchmark_symbol="VNINDEX",
        risk_free_rate=0.05
    )
    
    # Get full analysis
    results = analyzer.get_full_analysis()
    
    print("=" * 50)
    print("PORTFOLIO ANALYSIS RESULTS")
    print("=" * 50)
    
    print("\n📈 Performance Metrics:")
    print(f"  Time-Weighted Return (TWR): {results['performance']['twr']:.2f}%")
    print(f"  Money-Weighted Return (MWR): {results['performance']['mwr']:.2f}%")
    print(f"  Annualized Return: {results['performance'].get('annualized_return', 0):.2f}%")
    print(f"  Annualized Volatility: {results['performance'].get('annualized_volatility', 0):.2f}%")
    print(f"  Sharpe Ratio: {results['performance'].get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown: {results['performance'].get('max_drawdown', 0):.2f}%")
    
    print("\n📊 Benchmark Comparison:")
    print(f"  Alpha: {results['benchmark_comparison'].get('alpha', 0):.2f}%")
    print(f"  Beta: {results['benchmark_comparison'].get('beta', 0):.2f}")
    print(f"  Information Ratio: {results['benchmark_comparison'].get('information_ratio', 0):.2f}")
    print(f"  Tracking Error: {results['benchmark_comparison'].get('tracking_error', 0):.2f}%")
    
    print("\n🏷️ Stock Contributions:")
    for contrib in results['contributions']:
        print(f"  {contrib['symbol']}: {contrib['stock_return']:.2f}% return, "
              f"{contrib['contribution']:.2f}% contribution")
