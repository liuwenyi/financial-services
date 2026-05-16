# Ptrade量化策略：多策略融合系统
# 日期: 2026年5月16日
# 目标: 100万资金，年化25%，最大回撤<10%

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

# ==================== Ptrade导入（根据实际环境调整）====================
try:
    from ptrade import *
    from ptrade.data import *
    from ptrade.api import *
    from ptrade.const import *
    HAS_PTRADE = True
except ImportError:
    HAS_PTRADE = False
    print("警告: Ptrade模块未安装，将使用模拟模式")

# ==================== 全局参数配置 ====================
class StrategyConfig:
    # 基础参数
    INIT_CAPITAL = 1000000  # 初始资金
    MAX_DRAWDOWN_TARGET = 0.10  # 最大回撤目标
    ANNUAL_RETURN_TARGET = 0.25  # 年化收益目标
    
    # 标的池（指数ETF）
    ETF_POOL = [
        '510300.SH',  # 沪深300ETF
        '510500.SH',  # 中证500ETF
        '159845.SZ',  # 中证1000ETF
        '159915.SZ',  # 创业板ETF
        '588000.SH'   # 科创50ETF
    ]
    
    # 策略权重分配
    STRATEGY_WEIGHTS = {
        'main': 0.60,      # 主策略
        'reversal': 0.15,  # 辅助策略1
        'volatility': 0.15, # 辅助策略2
        'timing': 0.10     # 辅助策略3
    }
    
    # 交易成本
    COMMISSION_RATE = 0.0003  # 佣金率
    SLIPPAGE = 0.0002  # 滑点
    
    # 风险控制参数
    SINGLE_STOCK_MAX_WEIGHT = 0.30  # 单标的最大仓位
    MAX_EQUITY_WEIGHT = 0.90  # 最大权益仓位
    STOP_LOSS_SINGLE = -0.08  # 单标的止损
    STOP_LOSS_PORTFOLIO_DAY = -0.05  # 组合单日止损
    STOP_LOSS_PORTFOLIO_TOTAL = -0.10  # 组合最大回撤止损
    
    # 再平衡参数
    REBALANCE_WEEKDAY = 0  # 周一再平衡（0=周一）
    REBALANCE_DEVIATION = 0.05  # 权重偏离阈值

# ==================== 策略主类 ====================
class MultiStrategyQuant:
    def __init__(self):
        self.config = StrategyConfig()
        
        # 状态变量
        self.init_capital = self.config.INIT_CAPITAL
        self.current_capital = self.init_capital
        self.max_value = self.init_capital
        self.trade_count = 0
        self.win_trades = 0
        
        # 持仓记录
        self.positions = {}  # {标的: 数量}
        self.position_cost = {}  # {标的: 成本价}
        
        # 历史记录
        self.daily_returns = []
        self.portfolio_values = [self.init_capital]
        self.dates = []
        
        # 数据缓存
        self.price_cache = {}
        
    def initialize(self):
        """策略初始化"""
        if HAS_PTRADE:
            # Ptrade环境初始化
            set_commission(self.config.COMMISSION_RATE)
            set_slippage(self.config.SLIPPAGE)
            set_benchmark('000300.SH')
    
    def on_bar(self, context):
        """主策略逻辑（K线周期调用）"""
        current_dt = context.current_dt if HAS_PTRADE else datetime.now()
        self.dates.append(current_dt)
        
        # 检查风控条件
        if not self.check_risk_limits(context):
            return
        
        # 获取各策略信号
        main_signal = self.get_main_strategy_signal(context)
        reversal_signal = self.get_reversal_strategy_signal(context)
        volatility_signal = self.get_volatility_strategy_signal(context)
        timing_signal = self.get_timing_strategy_signal(context)
        
        # 合成最终信号
        final_weights = self.combine_strategy_signals(
            main_signal,
            reversal_signal,
            volatility_signal,
            timing_signal
        )
        
        # 检查是否需要再平衡
        is_rebalance_day = self.check_rebalance_day(current_dt)
        if is_rebalance_day or self.check_position_deviation(final_weights):
            self.execute_rebalance(context, final_weights)
        
        # 更新统计信息
        self.update_statistics(context)
    
    # ==================== 主策略：趋势+多因子 ====================
    def get_main_strategy_signal(self, context):
        """主策略：趋势强度 + 动量 + 波动率调整"""
        signals = {}
        prices = self.get_history_prices(self.config.ETF_POOL, 60)
        
        for etf in self.config.ETF_POOL:
            if etf not in prices:
                continue
                
            close_prices = prices[etf]
            
            # 计算因子
            ma10 = np.mean(close_prices[-10:])
            ma60 = np.mean(close_prices[-60:])
            trend = (ma10 - ma60) / ma60 if ma60 != 0 else 0
            
            momentum = (close_prices[-1] - close_prices[-20]) / close_prices[-20] if close_prices[-20] != 0 else 0
            
            returns = np.diff(close_prices) / close_prices[:-1]
            volatility = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0.20
            vol_adj = 0.20 / volatility if volatility > 0 else 1.0
            
            combined = (trend + momentum) * vol_adj
            
            if combined > 0.3:
                signals[etf] = 1.0
            elif combined > 0.1:
                signals[etf] = 0.5
            elif combined > -0.1:
                signals[etf] = 0.2
            else:
                signals[etf] = 0.0
        
        # 选择最强的2个ETF
        sorted_etfs = sorted(signals.items(), key=lambda x: x[1], reverse=True)[:2]
        final_signal = {}
        for etf, score in sorted_etfs:
            if score > 0:
                final_signal[etf] = 1.0 / len(sorted_etfs)
        
        return final_signal
    
    # ==================== 辅助策略1：动量反转 ====================
    def get_reversal_strategy_signal(self, context):
        """辅助策略1：RSI超卖超买 + KDJ金叉死叉"""
        signals = {}
        prices = self.get_history_prices(self.config.ETF_POOL, 20)
        
        for etf in self.config.ETF_POOL:
            if etf not in prices:
                continue
                
            close_prices = prices[etf]
            
            rsi = self.calculate_rsi(close_prices, 14)
            k, d, j = self.calculate_kdj(prices, etf)
            
            signal = 0
            if rsi < 30:
                signal += 1
            elif rsi > 70:
                signal -= 1
                
            if k > d and len(close_prices) > 1:
                signal += 0.5
            elif k < d:
                signal -= 0.5
            
            if signal >= 1:
                signals[etf] = 1.0
            elif signal <= -1:
                signals[etf] = 0.0
            else:
                signals[etf] = 0.5
        
        return signals
    
    # ==================== 辅助策略2：波动率控制 ====================
    def get_volatility_strategy_signal(self, context):
        """辅助策略2：波动率择时"""
        prices = self.get_history_prices(self.config.ETF_POOL, 20)
        market_vol = self.calculate_market_volatility(prices)
        
        if market_vol < 0.15:
            base_weight = 1.0
        elif market_vol < 0.25:
            base_weight = 0.7
        elif market_vol < 0.35:
            base_weight = 0.4
        else:
            base_weight = 0.1
        
        signals = {}
        weight_per_etf = base_weight / len(self.config.ETF_POOL)
        for etf in self.config.ETF_POOL:
            signals[etf] = weight_per_etf
        
        return signals
    
    # ==================== 辅助策略3：择时风控 ====================
    def get_timing_strategy_signal(self, context):
        """辅助策略3：市场状态判断"""
        market_strength = self.calculate_market_strength(context)
        
        if market_strength > 60:
            base_weight = 1.0
        elif market_strength > 40:
            base_weight = 0.7
        elif market_strength > 30:
            base_weight = 0.3
        else:
            base_weight = 0.0
        
        signals = {}
        weight_per_etf = base_weight / len(self.config.ETF_POOL)
        for etf in self.config.ETF_POOL:
            signals[etf] = weight_per_etf
        
        return signals
    
    # ==================== 策略融合 ====================
    def combine_strategy_signals(self, main_sig, reversal_sig, volatility_sig, timing_sig):
        """合成各策略信号"""
        final_weights = {}
        
        all_etfs = set(main_sig.keys()) | set(reversal_sig.keys()) | \
                   set(volatility_sig.keys()) | set(timing_sig.keys())
        
        for etf in all_etfs:
            m_sig = main_sig.get(etf, 0.0)
            r_sig = reversal_sig.get(etf, 0.0)
            v_sig = volatility_sig.get(etf, 0.0)
            t_sig = timing_sig.get(etf, 0.0)
            
            w = self.config.STRATEGY_WEIGHTS
            combined = w['main'] * m_sig + w['reversal'] * r_sig + \
                       w['volatility'] * v_sig + w['timing'] * t_sig
            
            final_weights[etf] = combined
        
        final_weights = self.normalize_weights(final_weights)
        final_weights = self.apply_constraints(final_weights)
        
        return final_weights
    
    def normalize_weights(self, weights):
        total = sum(weights.values())
        if total == 0:
            return weights
        
        return {etf: w/total for etf, w in weights.items()}
    
    def apply_constraints(self, weights):
        constrained = {}
        total_weight = 0
        
        for etf, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            actual_weight = min(w, self.config.SINGLE_STOCK_MAX_WEIGHT)
            constrained[etf] = actual_weight
            total_weight += actual_weight
        
        if total_weight > self.config.MAX_EQUITY_WEIGHT:
            scale = self.config.MAX_EQUITY_WEIGHT / total_weight
            constrained = {etf: w*scale for etf, w in constrained.items()}
        
        return constrained
    
    # ==================== 辅助函数 ====================
    def get_history_prices(self, securities, count):
        """获取历史价格（兼容Ptrade和模拟模式）"""
        if HAS_PTRADE:
            return self._get_ptrade_prices(securities, count)
        else:
            return self._get_simulated_prices(securities, count)
    
    def _get_ptrade_prices(self, securities, count):
        """Ptrade环境获取价格"""
        prices = {}
        for sec in securities:
            bars = get_bars(sec, count=count, unit='1d', fields=['close'])
            if bars is not None and len(bars) > 0:
                prices[sec] = bars['close'].values
        return prices
    
    def _get_simulated_prices(self, securities, count):
        """模拟模式生成价格"""
        import random
        np.random.seed(42)
        
        prices = {}
        for sec in securities:
            if sec not in self.price_cache:
                # 生成模拟价格序列
                base_price = random.uniform(2, 5)
                returns = np.random.normal(0.001, 0.02, count)
                price_series = base_price * np.cumprod(1 + returns)
                self.price_cache[sec] = price_series
            else:
                # 扩展价格序列
                last_price = self.price_cache[sec][-1]
                new_returns = np.random.normal(0.001, 0.02, 10)
                new_prices = last_price * np.cumprod(1 + new_returns)
                self.price_cache[sec] = np.concatenate([self.price_cache[sec][1:], new_prices])
            
            prices[sec] = self.price_cache[sec][-count:]
        
        return prices
    
    def calculate_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100 if avg_gain > 0 else 50
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def calculate_kdj(self, prices, etf, n=9, m1=3, m2=3):
        if etf not in prices:
            return 50, 50, 50
        
        close = prices[etf]
        if len(close) < n:
            return 50, 50, 50
        
        lowest_low = np.min(close[-n:])
        highest_high = np.max(close[-n:])
        
        if highest_high == lowest_low:
            rsv = 50
        else:
            rsv = (close[-1] - lowest_low) / (highest_high - lowest_low) * 100
        
        k = rsv
        d = k
        j = 3 * k - 2 * d
        
        return k, d, j
    
    def calculate_market_volatility(self, prices):
        returns_list = []
        for etf in self.config.ETF_POOL:
            if etf in prices:
                vals = prices[etf]
                if len(vals) >= 20:
                    rets = np.diff(vals) / vals[:-1]
                    returns_list.extend(rets[-20:])
        
        if returns_list:
            return np.std(returns_list) * np.sqrt(252)
        
        return 0.20
    
    def calculate_market_strength(self, context):
        # 简化的市场强度计算
        return 55  # 中性偏强
    
    def check_rebalance_day(self, dt):
        return dt.weekday() == self.config.REBALANCE_WEEKDAY
    
    def check_position_deviation(self, target_weights):
        current_value = self.calculate_portfolio_value()
        if current_value == 0:
            return True
        
        current_weights = {}
        for etf, amount in self.positions.items():
            if amount > 0:
                price = self.get_current_price(etf)
                current_weights[etf] = (amount * price) / current_value
        
        all_etfs = set(current_weights.keys()) | set(target_weights.keys())
        for etf in all_etfs:
            curr = current_weights.get(etf, 0)
            targ = target_weights.get(etf, 0)
            if abs(curr - targ) > self.config.REBALANCE_DEVIATION:
                return True
        
        return False
    
    def check_risk_limits(self, context):
        current_value = self.calculate_portfolio_value()
        
        if current_value > self.max_value:
            self.max_value = current_value
        
        drawdown = (self.max_value - current_value) / self.max_value
        if drawdown >= self.config.STOP_LOSS_PORTFOLIO_TOTAL:
            print(f"触发最大回撤止损！回撤: {drawdown:.2%}")
            self.close_all_positions(context)
            return False
        
        for etf, amount in self.positions.items():
            if amount > 0 and etf in self.position_cost:
                cost = self.position_cost[etf]
                current = self.get_current_price(etf)
                loss = (current - cost) / cost
                if loss <= self.config.STOP_LOSS_SINGLE:
                    print(f"触发单标的止损！{etf}, 亏损: {loss:.2%}")
                    self.order_target(etf, 0)
        
        return True
    
    def get_current_price(self, etf):
        if HAS_PTRADE:
            return get_current_price(etf)
        else:
            if etf in self.price_cache:
                return self.price_cache[etf][-1]
            return 1.0
    
    def calculate_portfolio_value(self):
        value = self.current_capital
        for etf, amount in self.positions.items():
            if amount > 0:
                price = self.get_current_price(etf)
                value += amount * price
        return value
    
    def close_all_positions(self, context):
        for etf, amount in list(self.positions.items()):
            if amount > 0:
                self.order_target(etf, 0)
        print("已平掉所有仓位")
    
    def order_target(self, etf, target_amount):
        current_amount = self.positions.get(etf, 0)
        diff = target_amount - current_amount
        
        if diff != 0:
            price = self.get_current_price(etf)
            cost = abs(diff) * price * (self.config.COMMISSION_RATE + self.config.SLIPPAGE)
            
            if diff > 0:
                self.current_capital -= diff * price + cost
                self.position_cost[etf] = price
            else:
                self.current_capital += abs(diff) * price - cost
                if etf in self.position_cost:
                    del self.position_cost[etf]
            
            if target_amount > 0:
                self.positions[etf] = target_amount
            else:
                if etf in self.positions:
                    del self.positions[etf]
    
    def execute_rebalance(self, context, target_weights):
        total_value = self.calculate_portfolio_value()
        
        for etf in list(self.positions.keys()):
            if self.positions[etf] > 0 and etf not in target_weights:
                self.order_target(etf, 0)
                print(f"卖出: {etf}")
        
        for etf, target_weight in target_weights.items():
            target_value = total_value * target_weight
            current_price = self.get_current_price(etf)
            
            if current_price > 0:
                target_amount = int(target_value / current_price)
                self.order_target(etf, target_amount)
                print(f"调整仓位: {etf}, 目标权重: {target_weight:.2%}")
        
        self.trade_count += 1
    
    def update_statistics(self, context):
        current_value = self.calculate_portfolio_value()
        self.portfolio_values.append(current_value)
        
        if len(self.portfolio_values) > 1:
            daily_return = (current_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
            self.daily_returns.append(daily_return)
    
    def analyze_performance(self):
        """分析策略绩效"""
        if not self.daily_returns:
            return {}
        
        values = np.array(self.portfolio_values)
        total_return = (values[-1] - self.init_capital) / self.init_capital
        n_days = len(self.daily_returns)
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        
        cumulative = values / self.init_capital
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        daily_returns = np.diff(values) / values[:-1]
        daily_vol = np.std(daily_returns)
        annual_vol = daily_vol * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        print(f"=== 策略绩效分析 ===")
        print(f"总收益率: {total_return:.2%}")
        print(f"年化收益率: {annual_return:.2%}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"夏普比率: {sharpe:.2f}")
        print(f"交易次数: {self.trade_count}")
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'trade_count': self.trade_count
        }

# ==================== 策略实例 ====================
strategy = MultiStrategyQuant()

# Ptrade框架函数
def initialize(context):
    strategy.initialize()

def handle_bar(context):
    strategy.on_bar(context)

# ==================== 模拟运行（用于测试）====================
def run_simulation(n_days=252):
    """模拟运行策略"""
    print(f"开始模拟运行，天数: {n_days}")
    
    strategy = MultiStrategyQuant()
    strategy.initialize()
    
    for i in range(n_days):
        # 模拟上下文
        class Context:
            pass
        context = Context()
        
        strategy.on_bar(context)
        
        if (i + 1) % 21 == 0:  # 每月输出
            value = strategy.calculate_portfolio_value()
            ret = (value - strategy.init_capital) / strategy.init_capital
            print(f"第{i+1}天, 收益: {ret:.2%}")
    
    performance = strategy.analyze_performance()
    return performance

if __name__ == "__main__":
    if not HAS_PTRADE:
        print("运行模拟模式...")
        performance = run_simulation(252)
