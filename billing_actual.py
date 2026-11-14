"""Observed account balances are not attributable per-task invoice charges."""
from decimal import Decimal,InvalidOperation
from datetime import datetime,timezone
import time

def account_observation(before,after):
    prior={(x['account'],x['currency']):x for x in before};changes=[]
    for row in after:
        old=prior.get((row['account'],row['currency']))
        if old:
            changes.append({'account':row['account'],'currency':row['currency'],'before':old['balance'],'after':row['balance'],
                'decrease':str(Decimal(old['balance'])-Decimal(row['balance']))})
    return {'task_actual_charge':None,'task_actual_status':'provider_task_receipt_unavailable',
        'attribution':'account_window_not_task_charge','observations':changes,'before':before,'after':after,
        'note':'账户窗口余额变化可能包含并发消费、充值或赠额变动，不能归因于本次研究；本轮精确实扣待账单对账。'}

class BalanceMeter:
    def __init__(self,clients):
        seen=set();self.accounts=[];self.before=[];self.errors=[]
        for client in clients.values():
            identity=getattr(client,'billing_account_identity',None)
            if identity and identity not in seen and callable(getattr(client,'balance_snapshot',None)):
                seen.add(identity);self.accounts.append(client)
    def capture(self):
        rows=[];deadline=time.monotonic()+2.5
        for i,client in enumerate(self.accounts[:4]):
            try:
                remaining=deadline-time.monotonic()
                if remaining<=0:break
                for row in client.balance_snapshot(timeout=remaining):
                    amount=Decimal(str(row['balance']))
                    if row['currency'] not in ('CNY','USD') or not amount.is_finite() or amount<0:continue
                    rows.append({'account':'account_'+str(i+1),'currency':row['currency'],'balance':str(amount),'observed_at':datetime.now(timezone.utc).isoformat()})
            except Exception:
                self.errors.append({'account':'account_'+str(i+1),'status':'balance_unavailable'})
        return rows
    def start(self):self.before=self.capture()
    def finish(self):
        result=account_observation(self.before,self.capture());result['errors']=self.errors
        return result
