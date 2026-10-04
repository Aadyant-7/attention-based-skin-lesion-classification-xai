"""Meaningful dual-metric convergence; raw checkpoint ranking stays independent."""
import numpy as np

POLICY = dict(version='meaningful_v2',max_epochs=50,minimum_epochs=25,
              accuracy_delta=.002,macro_f1_delta=.003,patience=12,
              plateau_after_epoch=30,plateau_window=10,lr_settle_epochs=3)

def stopping_status(history):
    state=dict(policy=POLICY,accuracy_reference=None,macro_f1_reference=None,
               last_accuracy_improvement_epoch=0,last_macro_f1_improvement_epoch=0,
               last_meaningful_epoch=0,stale=0,stop=False,reason='running')
    reductions=[]
    for row in history:
        epoch=int(row['epoch'])
        for key,column,delta in [('accuracy','val_accuracy',.002),('macro_f1','val_macro_f1',.003)]:
            value=float(row[column]);reference=state[key+'_reference']
            if reference is None or value-reference>=delta-1e-12:
                state[key+'_reference']=value
                state['last_'+key+'_improvement_epoch']=epoch
                state['last_meaningful_epoch']=epoch
        if float(row.get('lr_after',1))<float(row.get('backbone_lr',1))-1e-15:reductions.append(epoch)
    epoch=int(history[-1]['epoch']) if history else 0
    state['stale']=epoch-state['last_meaningful_epoch']
    state['accuracy_stale']=epoch-state['last_accuracy_improvement_epoch']
    state['macro_f1_stale']=epoch-state['last_macro_f1_improvement_epoch']
    state['lr_reduction_epochs']=reductions
    settled=bool(reductions) and epoch-reductions[-1]>=POLICY['lr_settle_epochs']
    recent=history[-10:];flat=False
    if len(recent)==10:
        flat=True
        for column,delta in [('val_accuracy',.002),('val_macro_f1',.003)]:
            values=np.array([float(r[column]) for r in recent])
            slope=float(np.polyfit(np.arange(10),values,1)[0])
            flat &= slope*9<delta and float(values[-3:].mean()-values[:3].mean())<delta
    state['flat_or_declining']=bool(flat);state['lr_settled']=settled
    if epoch>=50:state.update(stop=True,reason='epoch_cap')
    elif epoch>=25 and settled:
        if epoch>=30 and state['stale']>=10 and flat:state.update(stop=True,reason='late_meaningful_plateau')
        elif state['stale']>=12:state.update(stop=True,reason='meaningful_patience_exhausted')
    return state
