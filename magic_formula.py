import numpy as np
import pandas as pd
import datetime as dt

import traceback

def initialize(context):
    # 设置费率
    set_commission(PerTrade(buy_cost=0.00025, sell_cost=0.00125, min_cost=5))
    
    # 设置基准指数：沪深300指数 '000300.XSHG'
    set_benchmark('000300.XSHG')
    log.set_level('order', 'error')
    # 昨天
    g.yesterday = context.current_dt -dt.timedelta(1)
    g.stock_num = 30
    
    run_monthly(stock_trading, 2, time='open')

# 每三个月调仓换股
def stock_trading(context) :
    g.yesterday = context.current_dt -dt.timedelta(1)
    #if g.yesterday.month not in [1, 4, 7, 10]:
    #if g.yesterday.month not in [2, 5, 8, 11]:
    if g.yesterday.month not in [3, 6, 9, 12]:
        return
    
    buy_stocks = list(select_stocks(context, g.stock_num))
    dt_str = g.yesterday.strftime('%Y-%m-%d')
    #for stock in buy_stocks: print "#MF_LIST#%s %s" % (dt_str, stock)
    weight = get_weight(buy_stocks)

    # ------先清仓------
    for _stock in context.portfolio.positions:
        if _stock in buy_stocks:
            #buy_stocks.remove(_stock)
            continue
        order_target(str(_stock), 0)
    #print weight

    # ------再买股票了------
    # 单份金额
    #unit_money = context.portfolio.portfolio_value / g.stock_num
    for _stock in buy_stocks:
        _w = weight['weight'][_stock]
        #_w = 1.0 / g.stock_num
        unit_money = context.portfolio.portfolio_value * _w
        order_value(str(_stock), unit_money)

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
    return w
    
# 
def handle_data(context, data):
    pass


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
    df = df[map(lambda s: not s.startswith("ST") and not s.startswith("*ST") ,df.display_name)]
    
    return df.index.values
