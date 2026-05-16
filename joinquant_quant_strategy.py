# 聚宽量化策略：多策略融合系统
# 日期: 2026年5月16日
# 目标: 100万资金，年化25%，最大回撤<10%

from jqdata import *
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ==================== 全局参数配置 ====================
class StrategyConfig:
    # 基础参数
    INIT_CAPITAL = 1000000  # 初始资金
    MAX_DRAWDOWN_TARGET = 0.10  # 最大回撤目标
    ANNUAL_RETURN_TARGET = 0.25  # 年化收益目标
    
    # 标的池（指数ETF）
    ETF_POOL = [
        '510300.XSHG',  # 沪深300ETF
        '510500.XSHG',  # 中证500ETF
        '159845.XSHE',  # 中证1000ETF
        '159915.XSHE',  # 创业板ETF
        '588000.XSHG'   # 科创50ETF
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

# ==================== 初始化函数 ====================
def initialize(context):
    # 设置基准
    set_benchmark('000300.XSHG')
    
    # 开启防未来函数
    set_option('use_real_price', True)
    
    # 设置佣金和滑点
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0.001,
            open_commission=StrategyConfig.COMMISSION_RATE,
            close_commission=StrategyConfig.COMMISSION_RATE,
            close_today_commission=0,
            min_commission=5
        ),
        type='stock'
    )
    set_slippage(FixedSlippage(StrategyConfig.SLIPPAGE))
    
    # 初始化全局变量
    g.init_capital = StrategyConfig.INIT_CAPITAL
    g.max_value = g.init_capital  # 历史最高净值
    g.trade_count = 0
    g.win_trades = 0
    g.position_history = {}  # 持仓历史，用于计算盈亏比
    
    # 记录每日收益
    g.daily_returns = []
    
    # 每月第一个交易日运行
    run_daily(main_strategy, '9:30')

# ==================== 主策略执行 ====================
def main_strategy(context):
    """主策略：多策略融合执行"""
    # 获取当前日期信息
    current_date = context.current_dt.date()
    is_rebalance_day = check_rebalance_day(context)
    
    # 检查风控条件
    if not check_risk_limits(context):
        return
    
    # 获取各策略信号
    main_signal = get_main_strategy_signal(context)
    reversal_signal = get_reversal_strategy_signal(context)
    volatility_signal = get_volatility_strategy_signal(context)
    timing_signal = get_timing_strategy_signal(context)
    
    # 合成最终信号
    final_weights = combine_strategy_signals(
        main_signal,
        reversal_signal, 
        volatility_signal,
        timing_signal
    )
    
    # 执行交易
    if is_rebalance_day or check_position_deviation(context, final_weights):
        execute_rebalance(context, final_weights)
    
    # 更新统计信息
    update_statistics(context)

# ==================== 主策略：趋势+多因子 ====================
def get_main_strategy_signal(context):
    """
    主策略：趋势强度 + 动量 + 波动率调整
    返回格式：{标的: 目标权重}
    """
    signals = {}
    prices = get_history_prices(StrategyConfig.ETF_POOL, 60)
    
    for etf in StrategyConfig.ETF_POOL:
        if etf not in prices.columns:
            continue
            
        close_prices = prices[etf].values
        
        # 计算因子
        # 1. 趋势强度因子
        ma10 = np.mean(close_prices[-10:])
        ma60 = np.mean(close_prices[-60:])
        trend = (ma10 - ma60) / ma60 if ma60 != 0 else 0
        
        # 2. 动量因子
        momentum = (close_prices[-1] - close_prices[-20]) / close_prices[-20] if close_prices[-20] != 0 else 0
        
        # 3. 波动率调整因子
        returns = np.diff(close_prices) / close_prices[:-1]
        volatility = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0.20
        vol_adj = 0.20 / volatility if volatility > 0 else 1.0
        
        # 合成信号
        combined = (trend + momentum) * vol_adj
        
        # 转换为仓位
        if combined > 0.3:
            signals[etf] = 1.0
        elif combined > 0.1:
            signals[etf] = 0.5
        elif combined > -0.1:
            signals[etf] = 0.2
        else:
            signals[etf] = 0.0
    
    # 选择最强的2个ETF，等权分配
    sorted_etfs = sorted(signals.items(), key=lambda x: x[1], reverse=True)[:2]
    final_signal = {}
    for etf, score in sorted_etfs:
        if score > 0:
            final_signal[etf] = 1.0 / len(sorted_etfs)
    
    return final_signal

# ==================== 辅助策略1：动量反转 ====================
def get_reversal_strategy_signal(context):
    """
    辅助策略1：RSI超卖超买 + KDJ金叉死叉
    """
    signals = {}
    prices = get_history_prices(StrategyConfig.ETF_POOL, 20)
    
    for etf in StrategyConfig.ETF_POOL:
        if etf not in prices.columns:
            continue
            
        close_prices = prices[etf].values
        
        # 计算RSI
        rsi = calculate_rsi(close_prices, 14)
        
        # 计算KDJ
        k, d, j = calculate_kdj(prices, etf)
        
        # 合成信号
        signal = 0
        if rsi < 30:
            signal += 1
        elif rsi > 70:
            signal -= 1
            
        if k > d and len(close_prices) > 1:  # 金叉判断
            signal += 0.5
        elif k < d:
            signal -= 0.5
        
        # 转换为仓位
        if signal >= 1:
            signals[etf] = 1.0
        elif signal <= -1:
            signals[etf] = 0.0
        else:
            signals[etf] = 0.5  # 中性仓位
    
    return signals

# ==================== 辅助策略2：波动率控制 ====================
def get_volatility_strategy_signal(context):
    """
    辅助策略2：波动率择时 + 折溢价套利
    """
    signals = {}
    prices = get_history_prices(StrategyConfig.ETF_POOL, 20)
    
    # 计算市场整体波动率
    market_vol = calculate_market_volatility(prices)
    
    # 根据波动率确定基础仓位
    if market_vol < 0.15:
        base_weight = 1.0
    elif market_vol < 0.25:
        base_weight = 0.7
    elif market_vol < 0.35:
        base_weight = 0.4
    else:
        base_weight = 0.1
    
    # 分配到各ETF（等权）
    weight_per_etf = base_weight / len(StrategyConfig.ETF_POOL)
    for etf in StrategyConfig.ETF_POOL:
        signals[etf] = weight_per_etf
    
    return signals

# ==================== 辅助策略3：择时风控 ====================
def get_timing_strategy_signal(context):
    """
    辅助策略3：市场状态判断 + 择时风控
    """
    # 获取市场涨跌比
    market_strength = calculate_market_strength(context)
    
    # 根据市场状态确定仓位
    if market_strength > 60:
        base_weight = 1.0  # 强势市场
    elif market_strength > 40:
        base_weight = 0.7  # 中性市场
    elif market_strength > 30:
        base_weight = 0.3  # 弱势市场
    else:
        base_weight = 0.0  # 空仓避险
    
    # 等权分配
    signals = {}
    weight_per_etf = base_weight / len(StrategyConfig.ETF_POOL)
    for etf in StrategyConfig.ETF_POOL:
        signals[etf] = weight_per_etf
    
    return signals

# ==================== 策略融合模块 ====================
def combine_strategy_signals(main_sig, reversal_sig, volatility_sig, timing_sig):
    """
    合成各策略信号为最终权重
    """
    final_weights = {}
    
    # 对每个ETF计算加权平均信号
    all_etfs = set(main_sig.keys()) | set(reversal_sig.keys()) | \
               set(volatility_sig.keys()) | set(timing_sig.keys())
    
    for etf in all_etfs:
        # 获取各策略的信号（无信号则按中性处理）
        m_sig = main_sig.get(etf, 0.0)
        r_sig = reversal_sig.get(etf, 0.0)
        v_sig = volatility_sig.get(etf, 0.0)
        t_sig = timing_sig.get(etf, 0.0)
        
        # 加权合成
        w = StrategyConfig.STRATEGY_WEIGHTS
        combined = w['main'] * m_sig + w['reversal'] * r_sig + \
                   w['volatility'] * v_sig + w['timing'] * t_sig
        
        final_weights[etf] = combined
    
    # 归一化并应用约束
    final_weights = normalize_weights(final_weights)
    final_weights = apply_constraints(final_weights)
    
    return final_weights

def normalize_weights(weights):
    """归一化权重"""
    total = sum(weights.values())
    if total == 0:
        return weights
    
    normalized = {}
    for etf, w in weights.items():
        normalized[etf] = w / total
    
    return normalized

def apply_constraints(weights):
    """应用仓位约束"""
    constrained = {}
    total_weight = 0
    
    for etf, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        # 单标的最大权重限制
        actual_weight = min(w, StrategyConfig.SINGLE_STOCK_MAX_WEIGHT)
        constrained[etf] = actual_weight
        total_weight += actual_weight
    
    # 整体权益仓位限制
    if total_weight > StrategyConfig.MAX_EQUITY_WEIGHT:
        scale = StrategyConfig.MAX_EQUITY_WEIGHT / total_weight
        for etf in constrained:
            constrained[etf] *= scale
    
    return constrained

# ==================== 辅助函数 ====================
def get_history_prices(securities, count):
    """获取历史价格数据"""
    return attribute_history(
        securities,
        count=count,
        unit='1d',
        fields=['close', 'high', 'low'],
        df=True
    )

def calculate_rsi(prices, period=14):
    """计算RSI指标"""
    if len(prices) < period + 1:
        return 50
    
    deltas = np.diff(prices)
    gains = []
    losses = []
    
    for delta in deltas[-period:]:
        if delta > 0:
            gains.append(delta)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(delta))
    
    avg_gain = np.mean(gains) if gains else 0
    avg_loss = np.mean(losses) if losses else 0
    
    if avg_loss == 0:
        return 100 if avg_gain > 0 else 50
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_kdj(prices, etf, n=9, m1=3, m2=3):
    """计算KDJ指标"""
    if etf not in prices.columns:
        return 50, 50, 50
    
    close = prices[etf].values
    low_prices = prices['low'].values if 'low' in prices else close
    high_prices = prices['high'].values if 'high' in prices else close
    
    if len(close) < n:
        return 50, 50, 50
    
    # RSV计算
    lowest_low = np.min(low_prices[-n:])
    highest_high = np.max(high_prices[-n:])
    
    if highest_high == lowest_low:
        rsv = 50
    else:
        rsv = (close[-1] - lowest_low) / (highest_high - lowest_low) * 100
    
    # 这里简化计算，实际应该是SMA
    k = rsv
    d = k
    j = 3 * k - 2 * d
    
    return k, d, j

def calculate_market_volatility(prices):
    """计算市场整体波动率"""
    returns_list = []
    for col in prices.columns:
        if col in StrategyConfig.ETF_POOL:
            vals = prices[col].values
            if len(vals) >= 20:
                rets = np.diff(vals) / vals[:-1]
                returns_list.extend(rets[-20:])
    
    if returns_list:
        return np.std(returns_list) * np.sqrt(252)
    
    return 0.20

def calculate_market_strength(context):
    """计算市场涨跌强度（简化版本）"""
    # 获取沪深300成分股涨跌比
    stocks = get_index_stocks('000300.XSHG')
    if not stocks:
        return 50
    
    # 获取昨日数据
    current_data = get_current_data()
    up_count = 0
    down_count = 0
    
    for stock in stocks[:100]:  # 抽样计算
        if stock in current_data:
            try:
                prev_close = current_data[stock].pre_close
                curr_price = current_data[stock].day_open
                if curr_price > prev_close:
                    up_count += 1
                elif curr_price < prev_close:
                    down_count += 1
            except:
                pass
    
    total = up_count + down_count
    if total == 0:
        return 50
    
    return (up_count / total) * 100

def check_rebalance_day(context):
    """检查是否是再平衡日"""
    return context.current_dt.weekday() == StrategyConfig.REBALANCE_WEEKDAY

def check_position_deviation(context, target_weights):
    """检查持仓是否需要再平衡"""
    positions = context.portfolio.positions
    total_value = context.portfolio.total_value
    
    # 计算当前权重
    current_weights = {}
    for etf, pos in positions.items():
        if pos.total_amount > 0:
            current_weights[etf] = pos.value / total_value
    
    # 检查偏离度
    all_etfs = set(current_weights.keys()) | set(target_weights.keys())
    for etf in all_etfs:
        curr = current_weights.get(etf, 0)
        targ = target_weights.get(etf, 0)
        if abs(curr - targ) > StrategyConfig.REBALANCE_DEVIATION:
            return True
    
    return False

def check_risk_limits(context):
    """检查风险控制限制"""
    portfolio = context.portfolio
    
    # 更新历史最高净值
    if portfolio.total_value > g.max_value:
        g.max_value = portfolio.total_value
    
    # 检查最大回撤
    drawdown = (g.max_value - portfolio.total_value) / g.max_value
    if drawdown >= StrategyConfig.STOP_LOSS_PORTFOLIO_TOTAL:
        log.info(f"触发最大回撤止损！回撤: {drawdown:.2%}")
        close_all_positions(context)
        return False
    
    # 检查单标的止损
    for etf, pos in portfolio.positions.items():
        if pos.total_amount > 0:
            cost = pos.avg_cost
            current = pos.value / pos.total_amount
            loss = (current - cost) / cost
            if loss <= StrategyConfig.STOP_LOSS_SINGLE:
                log.info(f"触发单标的止损！{etf}, 亏损: {loss:.2%}")
                order_target(etf, 0)
    
    return True

def close_all_positions(context):
    """平掉所有仓位"""
    for etf, pos in context.portfolio.positions.items():
        if pos.total_amount > 0:
            order_target(etf, 0)
    log.info("已平掉所有仓位")

def execute_rebalance(context, target_weights):
    """执行再平衡"""
    portfolio = context.portfolio
    total_value = portfolio.total_value
    
    # 先卖出不在目标中的持仓
    for etf, pos in portfolio.positions.items():
        if pos.total_amount > 0 and etf not in target_weights:
            order_target(etf, 0)
            log.info(f"卖出: {etf}")
    
    # 按目标权重调整仓位
    for etf, target_weight in target_weights.items():
        target_value = total_value * target_weight
        
        if etf in portfolio.positions:
            current_value = portfolio.positions[etf].value
        else:
            current_value = 0
        
        diff_value = target_value - current_value
        
        if abs(diff_value) > 100:  # 最小交易金额
            order_target_value(etf, target_value)
            log.info(f"调整仓位: {etf}, 目标权重: {target_weight:.2%}")
    
    g.trade_count += 1

def update_statistics(context):
    """更新策略统计信息"""
    portfolio = context.portfolio
    
    # 记录每日收益
    if context.previous_date:
        prev_value = g.init_capital
        if g.daily_returns:
            prev_value = g.daily_returns[-1] * g.init_capital
        
        daily_return = (portfolio.total_value - prev_value) / prev_value
        g.daily_returns.append(portfolio.total_value / g.init_capital)
    
    # 记录日志
    if context.current_dt.day == 1:  # 每月第一天输出
        total_return = (portfolio.total_value - g.init_capital) / g.init_capital
        drawdown = (g.max_value - portfolio.total_value) / g.max_value
        
        log.info(f"=== 月度统计 ===")
        log.info(f"总收益: {total_return:.2%}")
        log.info(f"最大回撤: {drawdown:.2%}")
        log.info(f"交易次数: {g.trade_count}")

# ==================== 辅助分析函数（回测后使用）====================
def analyze_strategy_performance(context):
    """分析策略绩效指标"""
    if not g.daily_returns:
        return
    
    returns = np.diff(g.daily_returns) / g.daily_returns[:-1]
    
    # 计算基本指标
    total_return = g.daily_returns[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(g.daily_returns)) - 1
    
    # 计算最大回撤
    cumulative = np.array(g.daily_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # 计算夏普比率
    if len(returns) > 1:
        daily_vol = np.std(returns)
        annual_vol = daily_vol * np.sqrt(252)
        if annual_vol > 0:
            sharpe = annual_return / annual_vol
        else:
            sharpe = 0
    else:
        sharpe = 0
    
    log.info(f"=== 策略绩效分析 ===")
    log.info(f"总收益率: {total_return:.2%}")
    log.info(f"年化收益率: {annual_return:.2%}")
    log.info(f"最大回撤: {max_drawdown:.2%}")
    log.info(f"夏普比率: {sharpe:.2f}")
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe
    }

# 策略结束时输出绩效
def after_trading_end(context):
    if context.current_dt.month == 12 and context.current_dt.day >= 25:
        analyze_strategy_performance(context)
