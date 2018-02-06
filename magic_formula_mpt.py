import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import minimize

import traceback

def initialize(context):
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)

    # 设置费率
    set_commission(PerTrade(buy_cost=0.00025, sell_cost=0.00125, min_cost=5))
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    #set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')

    # 设置基准指数：沪深300指数 '000300.XSHG'
    set_benchmark('000300.XSHG')
    log.set_level('order', 'error')
    # 昨天
    g.yesterday = context.current_dt -dt.timedelta(1)
    g.stock_num = 30
    g.mpt_history_days = 252
    g.exp_ret = 0.2
    g.last_adjust_month = 0
    g.stocks_hold = []

    run_monthly(stock_trading_monthly, 2, time='open')
    #run_daily(stock_trading_daily, time='open')

# 每三个月调仓换股,其余时间调整持仓各股占比
def stock_trading_daily(context):
    g.yesterday = context.current_dt -dt.timedelta(1)
    # adjust_month = [1, 4, 7, 10]
    # adjust_month = [2, 5, 8, 11]
    adjust_month = [3, 6, 9, 12]
    cur_month = context.current_dt.month
    if cur_month in adjust_month and cur_month != g.last_adjust_month:
        g.stocks_hold = list(select_stocks(context, g.stock_num))
        g.last_adjust_month = cur_month
        # ------先清仓------
        for _stock in context.portfolio.positions:
            if _stock in g.stocks_hold: continue
            order_target(str(_stock), 0)
        #print weights
    if len(g.stocks_hold) < 1: return

    dt_str = g.yesterday.strftime('%Y-%m-%d')
    #for stock in g.stocks_hold: print "#MF_LIST#%s %s" % (dt_str, stock)
    #weights = get_weight(g.stocks_hold)
    weights = get_weight_mpt(g.stocks_hold)


    # ------再买股票------
    # 单份金额
    #unit_money = context.portfolio.portfolio_value / g.stock_num
    for _stock in g.stocks_hold:
        _w = weights[_stock]
        if _w < 0.00001:
            order_target(str(_stock), 0)
            continue
        #print "stock:%s, weight=%.2f" % (_stock, _w)
        #_w = 1.0 / g.stock_num
        unit_money = context.portfolio.portfolio_value * _w
        order_value(str(_stock), unit_money)


# 每三个月调仓换股
def stock_trading_monthly(context):
    g.yesterday = context.current_dt -dt.timedelta(1)
    #if g.yesterday.month not in [1, 4, 7, 10]:
    #if g.yesterday.month not in [2, 5, 8, 11]:
    if g.yesterday.month not in [3, 6, 9, 12]:
       return
    
    buy_stocks = list(select_stocks(context, g.stock_num))
    dt_str = g.yesterday.strftime('%Y-%m-%d')
    #for stock in buy_stocks: print "#MF_LIST#%s %s" % (dt_str, stock)
    #weights = get_weight(buy_stocks)
    weights = get_weight_mpt(buy_stocks)
    
    #cur_data = get_current_data()
    #for code in buy_stocks:
    #    t = cur_data[code]
    #    if t.is_st or t.paused:
    #        print ">>>>>> %s(%s) paused:%s" % (code, t.name, str(t.paused))

    # ------先清仓 或者 减仓------
    for _stock in context.portfolio.positions:
        ret = order_target(str(_stock), 0)
#        cash = context.portfolio.available_cash
#        _w = weights.get(_stock, 0)
#        if _w < 0.00001: _w = 0
#        unit_money = context.portfolio.portfolio_value * _w
#        long_pos = context.portfolio.long_positions.get(_stock, None)
#        value = 0 if long_pos == None else long_pos.value
#        if _stock in buy_stocks and unit_money >= value:
#            continue
#        print "--- %s, cash:%.2f, cur:%.2f, target:%.2f" % (_stock, cash, value, unit_money)
#        ret = order_target(str(_stock), unit_money)
        if ret == None:
            print "----- Failed to sell %s" % (_stock)

    # ------再买股票------
    # 单份金额
    #unit_money = context.portfolio.portfolio_value / g.stock_num
    idx = 0
    for _stock in buy_stocks:
        _w = weights[_stock]
        if _w < 0.00001:
            order_target(str(_stock), 0)
            continue
        info = get_security_info(_stock)
        idx = idx + 1
        #_w = 1.0 / g.stock_num
        unit_money = context.portfolio.portfolio_value * _w
        print "(%02d): %s(%s) %.2f(%.6f)" % (idx, _stock, info.display_name, unit_money, _w)
        #ret = order_value(_stock, unit_money)
        ret = order_target_value(_stock, unit_money)
        if ret == None:
            cash = context.portfolio.available_cash
            long_pos = None if _stock not in context.portfolio.long_positions else context.portfolio.long_positions[_stock]
            value = 0 if long_pos == None else long_pos.value
            print "+++++ Failed to buy %s(cur: %.2f, target: %.2f, cash:%.2f)" % (str(_stock),value,unit_money, cash)
    print "======================================\n\n"

def get_weight_mpt(stock_list):
    rets = get_history_ret(stock_list, g.mpt_history_days)
    #np_w = optimize(rets, 'sha')
    np_w = optimize(rets, 'ret', g.exp_ret)
    df = pd.DataFrame(np_w, stock_list)
    df.columns = ['weight']
    return df['weight']

def get_weight(stock_list):
    q = query(    
        valuation.code,                         # 股票代码
        valuation.circulating_market_cap,       # 流通市值(亿元)
    ).filter(
        valuation.code.in_(stock_list)
    )

    df = get_fundamentals(q, date=g.yesterday.strftime('%Y-%m-%d')).fillna(value=0).set_index('code')
    total = sum(df['circulating_market_cap'])
    w = df / total
    w.columns = ['weight']
    return w['weight']
    
# 
def handle_data(context, data):
    pass

def get_history_ret(stocks, days=30):
    data = history(days, "1d", "close", stocks, df=True, skip_paused=False, fq=None)
    rets = data.apply(lambda x: np.log(x/x.shift(1)))
    rets.columns = stocks
    return rets

def opt_ret(retvct, covmat, tgt):
    """
    Find the optimal weights for tgt expected return.
    """
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x},
            {'type': 'eq', 'fun': lambda x: get_stats(x, retvct, covmat)[0] - tgt})

    func = lambda x: get_stats(x, retvct, covmat)[1]

    x0 = [1./len(retvct) for x in retvct]
    return minimize(func, x0, constraints=cons, method='SLSQP').x
    

def opt_vol(retvct, covmat):
    """
    Find the weights for the portfolio with least variance.
    """
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x})

    func = lambda x: get_stats(x, retvct, covmat)[1]

    x0 = [1./len(retvct) for x in retvct]
    return minimize(func, x0, constraints=cons, method='SLSQP').x


def opt_sha(retvct, covmat):
    """
    Find the weights for the portfolio with maximized ret/vol (Sharpe Ratio).
    """
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x})

    func = lambda x: -get_stats(x, retvct, covmat)[2]

    x0 = [1./len(retvct) for x in retvct]
    return minimize(func, x0, constraints=cons, method='SLSQP').x


def optimize(rets, opt_type, tgt=None):
    retvct  = rets.mean().as_matrix()
    covmat  =  rets.cov().as_matrix()   

    if opt_type == 'ret':
        return opt_ret(retvct, covmat, tgt)
    elif opt_type == 'vol':
        return opt_vol(retvct, covmat)
    elif opt_type == 'sha':
        return opt_sha(retvct, covmat)

def get_stats(weights, retvct, covmat):
    """
    Compute portfolio anualized returns, volatility and ret/vol.
    """
    ret = np.dot(retvct, weights)*252
    std = np.sqrt(np.dot(weights, np.dot(covmat*252, weights)))
    return [ret, std, ret/std]


def get_stats2(rets, weights):
    """
    Same as get_stats, but takes DataFrame with returns as input.
    """
    retvct = rets.mean().as_matrix()
    covmat = rets.cov().as_matrix()   

    return get_stats(weights, retvct, covmat)



# 根据魔法公式计算的结果进行选股
def select_stocks(context,num):
    ROC,EY = cal_magic_formula(context)

    # 按ROC 和 EY 构建表格
    ROC_EY = pd.DataFrame({'ROC': ROC,'EY': EY})

    # 对 ROC进行降序排序, 记录序号
    ROC_EY = ROC_EY.sort('ROC',ascending=False)
    idx = pd.Series(np.arange(1,len(ROC)+1), index=ROC_EY['ROC'].index.values)
    ROC_I = pd.DataFrame({'ROC_I': idx})
    ROC_EY = pd.concat([ROC_EY, ROC_I], axis=1)

    # 对 EY进行降序排序, 记录序号
    ROC_EY = ROC_EY.sort('EY',ascending=False)
    idx = pd.Series(np.arange(1,len(EY)+1), index=ROC_EY['EY'].index.values)
    EY_I = pd.DataFrame({'EY_I': idx})
    ROC_EY = pd.concat([ROC_EY, EY_I], axis=1)

    # 对序号求和，并记录之
    roci = ROC_EY['ROC_I']
    eyi = ROC_EY['EY_I']
    idx = roci + eyi
    SUM_I = pd.DataFrame({'SUM_I': idx})
    ROC_EY = pd.concat([ROC_EY, SUM_I], axis=1)

    # 按序号和，进行升序排序，然后选出排名靠前的20只股票
    ROC_EY = ROC_EY.sort('SUM_I')
    ROC_EY = ROC_EY.head(num)
    
    return ROC_EY.index.values
    
# 计算魔法公式
def cal_magic_formula(context):
    stocks = filter_stocks(context)
    
    q = query(    
        valuation.code,                         # 股票代码
        valuation.market_cap,                   # 总市值(亿元)
        valuation.circulating_market_cap,       # 流通市值(亿元)
        income.net_profit,                      # 净利润(元)
        income.financial_expense,               # 财务费用(元)
        income.income_tax_expense,              # 所得税费用(元)
        balance.fixed_assets,                   # 固定资产(元)
        balance.construction_materials,         # 工程物资(元)
        balance.constru_in_process,             # 在建工程(元)
        balance.fixed_assets_liquidation,       # 固定资产清理(元)
        balance.total_current_assets,           # 流动资产合计(元)
        balance.total_current_liability,        # 流动负债合计(元)
        balance.total_liability,                # 负债合计(元)
        balance.total_sheet_owner_equities,     # 负债和股东权益合计
        balance.dividend_payable,               # 应付股利(元)
        cash_flow.cash_and_equivalents_at_end   # 期末现金及现金等价物余额(元)
    ).filter(
        income.net_profit > 0,
        valuation.market_cap > 50,
        valuation.code.in_(stocks)
    )

    df = get_fundamentals(q, date=g.yesterday.strftime('%Y-%m-%d')).fillna(value=0).set_index('code')
    
    # 息税前利润(EBIT) = 净利润 + 财务费用 + 所得税费用
    NP = df['net_profit']
    FE = df['financial_expense']
    TE = df['income_tax_expense']
    EBIT = NP + FE + TE

    # 固定资产净额(Net Fixed Assets) = 固定资产 - 工程物资 - 在建工程 - 固定资产清理
    FA = df['fixed_assets']
    CM = df['construction_materials']
    CP = df['constru_in_process']
    FAL = df['fixed_assets_liquidation']
    NFA = FA - CM - CP - FAL

    # 净营运资本(Net Working Capital)= 流动资产合计－流动负债合计
    TCA = df['total_current_assets']
    TCL = df['total_current_liability']
    NWC = TCA - TCL
    
    # 全部投入资本（MRQ）= 负债 + 股东权益 + 应付股利
    TSOE = df['total_sheet_owner_equities']
    DP = df['dividend_payable']
    MRQ = TSOE + DP

    # 企业价值(Enterprise Value) = 总市值 + 负债合计 – 期末现金及现金等价物余额
    MC = df['market_cap']*100000000
    TL = df['total_liability']
    TC = df['cash_and_equivalents_at_end']
    EV = MC + TL - TC

    # Net Working Capital + Net Fixed Assets
    NCA = NWC + NFA

    # 剔除 NCA 和 EV 非正的股票
    tmp = set(df.index.values)-set(EBIT[EBIT<=0].index.values)-set(EV[EV<=0].index.values)-set(NCA[NCA<=0].index.values)
    EBIT = EBIT[tmp]
    NCA = NCA[tmp]
    MRQ = MRQ[tmp]
    EV = EV[tmp]

    # 计算魔法公式
    ROC = EBIT / NCA
    ROIC = EBIT / MRQ
    EY = EBIT / EV
    
    return [ROC,EY]
    #return [ROIC,EY]
    
# 从所有A股中过滤掉：金融服务，公用事业，创业板，新股次新股，ST  
def filter_stocks(context):
    # 获取所有A股
    df = get_all_securities('stock')
    # 剔除金融服务，公用事业
    #tmp = set(df.index.values)-set(get_industry_stocks('J66'))\
    #        -set(get_industry_stocks('J67'))\
    #        -set(get_industry_stocks('J68'))\
    #        -set(get_industry_stocks('J69'))\
    #        -set(get_industry_stocks('N78')) 
    tmp = set(df.index.values) - set(get_industry_stocks('D44', g.yesterday))\
            -set(get_industry_stocks('D45', g.yesterday))\
            -set(get_industry_stocks('D46', g.yesterday))\
            -set(get_industry_stocks('J66', g.yesterday))\
            -set(get_industry_stocks('J67', g.yesterday))\
            -set(get_industry_stocks('J68', g.yesterday))\
            -set(get_industry_stocks('J69', g.yesterday))\
            -set(get_industry_stocks('N78', g.yesterday)) 
    tmp = np.array(list(tmp))
    df = df.select(lambda code: code in tmp)
    #  剔除创业板
    df = df.select(lambda code: not code.startswith('300'))
    #  新股次新股
    one_year = dt.timedelta(365)
    df = df[df.start_date < g.yesterday.date() - one_year]
    # 剔除ST
    #df = df[map(lambda s: not s.startswith("ST") and not s.startswith("*ST") ,df.display_name)]
    
    cur_data = get_current_data()
    # 剔除ST、停牌、暂停交易股票
    df = df.select(lambda code: not cur_data[code].is_st and not cur_data[code].paused)
    #for code in df.index.values:
    #    t = cur_data[code]
    #    if t.is_st or t.paused:
    #        print ">>>>>> %s(%s) paused:%s" % (code, t.name, str(t.paused))
    
    return df.index.values
