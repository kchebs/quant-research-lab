from agent.TradingAgent import TradingAgent
import pandas as ps
import numpy as my
from contributed_traders.util import get_file

# Simple Trading Agent was used as the framework to learn and create this agent; if leverage was used, it was accidental and unknowingly

class MomentumInventoryAgent(TradingAgent):

    def __init__(self, id, name, type, symbol, starting_cash,
                 min_size, max_size, wake_up_freq='60s',
                 log_orders=False, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.end = False
        self.symbol = symbol
        self.max_size = max_size
        self.wake_up_freq = wake_up_freq
        self.min_size = min_size
        self.size = self.random_state.randint(self.min_size, self.max_size)
        self.log_orders = log_orders
        self.md_lst, self.avg_win1s, self.avg_win2s = [], [], []
        self.state = "AWAITING_WAKEUP"
        self.starting_cash = starting_cash
        self.holdings = {'CASH': starting_cash}
        self.count, self.pending, self.pause = 0, 0, 0
        self.dpth = 10
        self.prcVol = 100
        self.Sprd = ps.DataFrame([50, 51])

    # noinspection PyMethodMayBeStatic
    def author(self):
        return "momentum_inventory"

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        with open(get_file('momentum_inventory_agent/myagent.cfg'), 'r') as cfg:
            self.wnd1, self.wnd2 = [int(c) for c in cfg.readline().split()]

    def wakeup(self, currentTime):
        if not super().wakeup(currentTime): return
        if 0 == self.pending and self.pause <= 0:
            self.cxOrdrs()
            try:
                self.sprd_std = self.Sprd.std()[0]
            except:
                self.sprd_std = 50
            if not self.end:
                self.state = 'AWAITING_SPREAD'
            self.getCurrentSpread(self.symbol, depth=self.dpth)
        else:
            self.state = 'AWAITING_WAKEUP'
            self.pause -= 2
            self.setWakeup(currentTime + self.getWakeFrequency())
        if self.end:
            self.cxOrdrs()
            self.dmp_shr()

    def cxOrdrs(self):
        for _, prchs in self.orders.items():
            self.cancelOrder(prchs)

    def agentname(self):
        return 'MomentumInventoryAgent'

    def dmp_shr(self):
        if 0 == len(self.orders) and self.symbol in self.holdings:
            bd, _, ak, _ = self.getKnownBidAsk(self.symbol)
            ordr_sz = abs(self.holdings[self.symbol])
            if bd:
                self.placeLimitOrder(self.symbol, quantity=ordr_sz, is_buy_order=False, limit_price=0)
            elif ak:
                self.placeLimitOrder(self.symbol, quantity=ordr_sz, is_buy_order=True, limit_price=round(ak*2))

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if self.end:
            if 'ORDER_EXECUTED' == msg.body['msg']:
                try:
                    if 0 != self.holdings[self.symbol]:
                        self.cxOrdrs()
                        self.dmp_shr()
                        self.state = 'AWAITING_WAKEUP'
                except:
                    pass
        elif 'QUERY_SPREAD' == msg.body['msg'] and self.state == 'AWAITING_SPREAD':
            dtime = (self.mkt_close-currentTime)/my.timedelta64(1, 'm')
            if 25 > dtime:
                self.dmp_shr()
            else:
                bd, _, ak, _ = self.getKnownBidAsk(self.symbol)
                if bd and ak:
                    self.md_lst.append((ak+bd)/2)
                    if self.wnd2 < len(self.md_lst): self.avg_win2s.append(
                        ps.Series(self.md_lst).ewm(span=self.wnd2).mean().values[-1].round(3))
                    if self.wnd1 < len(self.md_lst): self.avg_win1s.append(
                        ps.Series(self.md_lst).ewm(span=self.wnd1).mean().values[-1].round(3))
                    if 0 == len(self.orders) and 0 < len(self.avg_win1s) and 0 < len(self.avg_win2s):
                        if self.avg_win2s[-1] < self.avg_win1s[-1] and (self.size * ak) <= self.holdings['CASH']:
                            self.action = 'By'
                            self.calcOrder()
                            if 5 > dtime:
                                self.end = True
                                self.dmp_shr()
                            self.state = 'AWAITING_WAKEUP'
                            self.setWakeup(currentTime + self.getWakeFrequency())
                        elif self.symbol in self.holdings and self.holdings[self.symbol] > 0:
                            self.action = 'Sll'
                            self.calcOrder()
                            if 5 > dtime:
                                self.end = True
                                self.dmp_shr()
                            self.state = 'AWAITING_WAKEUP'
                            self.setWakeup(currentTime + self.getWakeFrequency())
            self.setWakeup(currentTime + self.getWakeFrequency())
            self.state = 'AWAITING_WAKEUP'
        elif self.state == 'AWAITING_WAKEUP' and msg.body['msg'] == 'ORDER_EXECUTED':
            if len(self.orders) > 0 and self.pause == 0:
                self.pause = 2
            else:
                self.pause = 0
        elif msg.body['msg'] == 'ORDER_ACCEPTED':
            self.count += 1
            self.pending -= 1
            
    def number_of_counting(self):
        return self.count

    def calcOrder(self):
        bd, ak = self.getKnownBidAsk(self.symbol, best=False)
        if bd and ak:
            volb_sm, bid_sm, vola_sm, ask_sm = 0, 0, 0, 0
            try:
                for d in range(self.dpth):
                    if volb_sm < self.prcVol < volb_sm + bd[d][1]:
                        bid_sm += (self.prcVol - volb_sm) * bd[d][0]
                        volb_sm = self.prcVol
                    elif self.prcVol > volb_sm:
                        bid_sm += bd[d][1] * bd[d][0]
                        volb_sm += bd[d][1]
                    if vola_sm < self.prcVol < vola_sm + ak[d][1]:
                        ask_sm += ak[d][0]*(self.prcVol - vola_sm)
                        vola_sm = self.prcVol
                    elif self.prcVol > vola_sm:
                        vola_sm += ak[d][1]
                        ask_sm += ak[d][0]*ak[d][1]
                    if vola_sm == self.prcVol and self.prcVol == volb_sm:
                        break
                if volb_sm == vola_sm and volb_sm == self.prcVol:
                    bmid, amid = bid_sm/self.prcVol, ask_sm/self.prcVol
                    mmid = (bmid+amid)/2
                    volb = int(my.floor(max(0, self.holdings['CASH']/mmid)))
                    try:
                        vola = int(my.floor(max(0,(self.holdings['CASH'] - 2*abs(min(0,self.holdings[self.symbol]*amid)))/mmid + max(0,self.holdings[self.symbol])*2)))
                    except:
                        vola = int(my.floor(max(0,self.holdings['CASH']/mmid)))
                    mPrc = volb/(vola+volb) * self.sprd_std/7 - self.sprd_std/14 + mmid
                    aPrc, bPrc = int(my.ceil(max(self.sprd_std/1.5 + mPrc, -1+amid))), int(my.floor(min(-self.sprd_std/1.5 + mPrc, 1+bmid)))
                    if self.action == 'By' and 0 < volb:
                        self.pending += 1
                        self.placeLimitOrder(self.symbol, volb, is_buy_order=True, limit_price=bPrc)
                    elif self.action == 'Sll' and 0 < vola:
                        self.pending += 1
                        self.placeLimitOrder(self.symbol, vola, is_buy_order=False, limit_price=aPrc)
                    self.Sprd = self.Sprd.append([-bmid+amid], ignore_index=True)
            except:
                pass

    def getWakeFrequency(self):
        return ps.Timedelta(self.wake_up_freq)

    def getHoldings(self, tckr):
        if tckr in self.holdings: return self.holdings[tckr]
        return 0