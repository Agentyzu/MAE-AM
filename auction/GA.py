import numpy as np


def GA(b, pctr, pos_norm, q, constant, len_content, max_len):
    b,pctr,pos,lengths=(np.asarray(v,dtype=float) for v in (b,pctr,pos_norm,len_content))
    if b.ndim!=1 or not len(b) or pctr.shape!=b.shape or lengths.shape!=b.shape:
        raise ValueError("b, pctr and creative lengths must be nonempty equal-length vectors")
    if isinstance(q,(bool,np.bool_)) or not isinstance(q,(int,np.integer)) or not 1<=q<=len(b):
        raise ValueError("q must be an integer between 1 and the number of advertisers")
    if not np.all(np.isfinite(b)) or np.any(b<0):
        raise ValueError("bids must be finite and nonnegative")
    if not np.all(np.isfinite(pctr)) or np.any(pctr<=0) or np.any(pctr>1):
        raise ValueError("base CTRs must lie in (0,1]")
    if (pos.ndim!=1 or len(pos)<q or not np.all(np.isfinite(pos[:q]))
            or np.any(pos[:q]<=0) or np.any(np.diff(pos[:q])>0)):
        raise ValueError("position factors must be positive and non-increasing")
    if (not np.isfinite(max_len) or max_len<1 or max_len!=int(max_len)
            or not np.all(np.isfinite(lengths)) or np.any(lengths<0) or np.any(lengths!=np.floor(lengths))):
        raise ValueError("creative lengths must be nonnegative integer counts and budget a positive integer")
    if not np.isfinite(constant):
        raise ValueError("constant must be finite")
    scores=b*pctr
    ranking=np.argsort(-scores)
    selected=[]; allocated=[]; remaining=int(max_len)
    for i in ranking:
        length=min(int(lengths[i]),remaining)
        if length>0:
            selected.append(int(i)); allocated.append(length); remaining-=length
            if len(selected)==q or remaining==0:
                break
    chosen=set(selected)
    sigma=np.array(selected+[int(i) for i in ranking if i not in chosen],dtype=int)
    count=len(selected)
    prominence,payments,utilities=(np.zeros(q) for _ in range(3))
    prominence[:count]=np.asarray(allocated,dtype=float)/max_len
    weighted=scores[sigma[:q]]*pos[:q]
    if count==1:
        payments[0]=scores[sigma[1]] if len(sigma)>1 else 0.
    elif count>1:
        for k in range(count):
            other=float(np.sum(weighted)-weighted[k])
            if other>0:
                t=weighted[k]/other
                payments[k]=(other/(pctr[sigma[k]]*pos[k]))*(np.log1p(t)-t/(1+t))
            # Zero competing scores give zero payment.
    utilities[:count]=b[sigma[:count]]*prominence[:count]-payments[:count]
    welfare=float(np.sum(weighted[:count]*(np.log(prominence[:count])+constant)))
    return sigma,prominence,payments,utilities,welfare
