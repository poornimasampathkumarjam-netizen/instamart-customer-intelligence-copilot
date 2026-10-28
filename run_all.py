"""Reproducible end-to-end pipeline for the synthetic Instamart capstone."""
from pathlib import Path
import os, re, json, time, random, sqlite3, hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import mock_llm

ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
_RUNTIME={}
for d in ['data/raw','data/clean','data/policies','outputs','state','sql']:(ROOT/d).mkdir(parents=True,exist_ok=True)
random.seed(42); np.random.seed(42)
def dump_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')

def part_a():
    c=pd.read_csv('data/raw/customers.csv',keep_default_na=False); p=pd.read_csv('data/raw/products.csv'); o=pd.read_csv('data/raw/orders.csv',keep_default_na=False)
    # Customer cleaning: canonical cities, genders, valid age then city median transform.
    c['city']=c.city.astype(str).str.strip().str.title()
    c['gender']=c.gender.astype(str).str.strip().str.lower().map({'male':'male','m':'male','female':'female','f':'female'}).fillna('unknown')
    c['age']=pd.to_numeric(c.age,errors='coerce'); c.loc[~c.age.between(10,100),'age']=np.nan
    c['age']=c['age'].fillna(c.groupby('city').age.transform('median')).round().astype(int)
    c['signup_date']=pd.to_datetime(c.signup_date).dt.strftime('%Y-%m-%d')
    # Orders: exact dupes then latest status record per id.
    o['order_ts']=pd.to_datetime(o.order_ts)
    o=o.drop_duplicates().sort_values('order_ts').drop_duplicates('order_id',keep='last').copy()
    o['quantity']=pd.to_numeric(o.quantity,errors='coerce'); o['discount_pct']=pd.to_numeric(o.discount_pct,errors='coerce')
    o['order_amount']=pd.to_numeric(o.order_amount,errors='coerce')
    o=o.merge(p[['product_id','price_inr','category']],on='product_id',how='left',validate='many_to_one')
    mask=o.order_amount.isna(); o.loc[mask,'order_amount']=(o.loc[mask,'price_inr']*o.loc[mask,'quantity']*(1-o.loc[mask,'discount_pct']/100)).round(2)
    o['delivery_minutes']=pd.to_numeric(o.delivery_minutes,errors='coerce'); o.loc[o.delivery_minutes<0,'delivery_minutes']=np.nan
    done=o.status.eq('completed'); o.loc[done,'delivery_minutes']=o.loc[done,'delivery_minutes'].fillna(o.loc[done].groupby('category').delivery_minutes.transform('median'))
    o['order_ts']=o.order_ts.dt.strftime('%Y-%m-%d %H:%M:%S')
    o=o[['order_id','customer_id','product_id','quantity','order_ts','status','discount_pct','order_amount','delivery_minutes']]
    c.to_csv('data/clean/customers.csv',index=False); p.to_csv('data/clean/products.csv',index=False); o.to_csv('data/clean/orders.csv',index=False)
    with sqlite3.connect('data/instamart.db') as db:
        c.to_sql('customers',db,index=False,if_exists='replace'); p.to_sql('products',db,index=False,if_exists='replace'); o.to_sql('orders',db,index=False,if_exists='replace')
    assert len(c)==1200 and len(p)==40 and len(o)==12309
    assert not o.loc[o.status.eq('completed'),'delivery_minutes'].isna().any()
    assert o.loc[o.status.eq('completed'),'order_amount'].sum().round(2)==2998668.80
    print('Part A checkpoints:',len(c),len(p),len(o),o.status.value_counts().to_dict(),'completed revenue',round(o.loc[o.status.eq('completed'),'order_amount'].sum(),2))
    # Query all seven saved SQL statements using split comment markers.
    sql=Path('sql/queries.sql').read_text(); queries=re.findall(r'-- Q\d+\n(.*?);',sql,re.S)
    query_results=[]
    for i,q in enumerate(queries,1):
        with sqlite3.connect('data/instamart.db') as db: df=pd.read_sql_query(q,db)
        query_results.append(df); print(f'Q{i}: {len(df)} rows')
    # Independent pandas cross-checks and completed-order exploratory statistics.
    completed=o[o.status.eq('completed')].merge(c[['customer_id','city']],on='customer_id').merge(p[['product_id','category']],on='product_id')
    pivot=pd.pivot_table(completed,index='city',columns='category',values='order_amount',aggfunc='sum',margins=True,margins_name='All',fill_value=0)
    city_stats=completed.groupby('city').agg(sum=('order_amount','sum'),mean=('order_amount','mean'),count=('order_amount','count')).reset_index()
    amount=completed.order_amount; q1=float(amount.quantile(.25)); q3=float(amount.quantile(.75)); iqr=q3-q1
    corr=completed[['order_amount','discount_pct','quantity','delivery_minutes']].corr()
    pairs=[(abs(float(corr.loc[a,b])),a,b,float(corr.loc[a,b])) for i,a in enumerate(corr.columns) for b in corr.columns[i+1:]]
    strongest=max(pairs)
    eda={'pandas_pivot_grand_total_inr':round(float(pivot.loc['All','All']),2),
         'city_stats':city_stats.round(2).to_dict('records'),
         'completed_order_amount':{'mean':float(amount.mean()),'median':float(amount.median()),'skewness':float(amount.skew()),
           'q1':q1,'q3':q3,'lower_fence':float(q1-1.5*iqr),'upper_fence':float(q3+1.5*iqr),
           'outliers_below':int((amount<q1-1.5*iqr).sum()),'outliers_above':int((amount>q3+1.5*iqr).sum()),
           'strongest_abs_correlation':{'feature_1':strongest[1],'feature_2':strongest[2],'correlation':strongest[3]}},
         'sql_row_counts':{f'Q{i+1}':int(len(df)) for i,df in enumerate(query_results)},
         'q2_bengaluru_electronics_inr':round(float(query_results[1].query("city == 'Bengaluru' and category == 'Electronics'").revenue.iloc[0]),2),
         'q5_average_completed_spend_inr':round(float(o[o.status.eq('completed')].groupby('customer_id').order_amount.sum().mean()),2),
         'q6_rank_not_row_number':int((query_results[5].rnk!=query_results[5].row_num).sum())}
    Path('outputs/eda_summary.json').write_text(json.dumps(eda,indent=2),encoding='utf-8')
    print('Pandas pivot grand total',eda['pandas_pivot_grand_total_inr'],'city stats',city_stats.round(2).to_dict('records'))
    return c,p,o

def part_b(c,p,o):
    x=o.copy(); x['order_ts']=pd.to_datetime(x.order_ts); completed=x[x.status.eq('completed')].copy()
    snap=pd.Timestamp('2025-10-01')
    rfm=completed.groupby('customer_id').agg(last=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index()
    rfm['recency']=(snap-rfm['last']).dt.days; rfm=rfm[['customer_id','recency','frequency','monetary']]
    cols=['recency','frequency','monetary']; scaler=StandardScaler(); Z=scaler.fit_transform(rfm[cols])
    scores=[]
    for k in range(2,9):
        km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(Z); scores.append({'k':k,'inertia':km.inertia_,'silhouette':silhouette_score(Z,km.labels_)})
    pd.DataFrame(scores).to_csv('outputs/kmeans_scores.csv',index=False)
    k=4; km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(Z); rfm['segment_id']=km.labels_
    prof=rfm.groupby('segment_id').agg(size=('customer_id','size'),recency=('recency','mean'),frequency=('frequency','mean'),monetary=('monetary','mean')).reset_index()
    names={}
    premium=int(prof.loc[prof.monetary.idxmax(),'segment_id'])
    for _,row in prof.iterrows():
        sid=int(row.segment_id)
        if sid==premium: name='Premium High-Value'
        elif row.frequency>=rfm.frequency.quantile(.65): name='Frequent Essentials'
        elif row.recency>=rfm.recency.quantile(.65): name='At-Risk / Cooling'
        else: name='Steady Value'
        names[sid]=name
    rfm['segment_name']=rfm.segment_id.map(names); rfm['is_premium']=rfm.segment_id.eq(premium)
    rfm.to_csv('outputs/customer_segments.csv',index=False)
    pca=PCA(n_components=2).fit(Z); print('PCA variance',pca.explained_variance_ratio_.round(3).tolist()); print('Segments',prof.round(2).to_dict('records'))
    # Lapse cohort/features strictly before 2025-09-01.
    cutoff=pd.Timestamp('2025-09-01'); x['date']=pd.to_datetime(x.order_ts); pre=x[(x.date<cutoff)]
    pre_done=pre[pre.status.eq('completed')]; cohort=pre_done.customer_id.unique(); sept=completed[(completed.order_ts>=cutoff)&(completed.order_ts<pd.Timestamp('2025-10-01'))].customer_id.unique()
    # Vectorised per-customer aggregates avoid repeatedly scanning the full order
    # table once per customer (which can take minutes on constrained machines).
    completed_agg=pre_done.groupby('customer_id').agg(
        last_order=('date','max'), frequency=('order_id','nunique'),
        monetary=('order_amount','sum'), avg_discount=('discount_pct','mean'),
        avg_delivery=('delivery_minutes','mean'))
    final_agg=pre.assign(is_final=pre.status.isin(['completed','cancelled']),
                         is_cancelled=pre.status.eq('cancelled')).groupby('customer_id').agg(
        final_orders=('is_final','sum'), cancelled=('is_cancelled','sum'))
    final_agg['cancel_rate']=final_agg.cancelled / final_agg.final_orders.clip(lower=1)
    fav_counts=(pre_done.merge(p[['product_id','category']],on='product_id',how='left')
                .groupby(['customer_id','category']).size().rename('n').reset_index()
                .sort_values(['customer_id','n','category'],ascending=[True,False,True])
                .drop_duplicates('customer_id').set_index('customer_id')['category'])
    cust_index=c.set_index('customer_id')
    f=completed_agg.loc[list(cohort)].copy()
    f['recency']=(cutoff-f.pop('last_order')).dt.days
    f=f.join(final_agg[['cancel_rate']]).join(fav_counts.rename('fav_category'))
    f['city']=cust_index.loc[f.index,'city']
    f['age']=cust_index.loc[f.index,'age']
    f['gender']=cust_index.loc[f.index,'gender']
    f['tenure']=(cutoff-pd.to_datetime(cust_index.loc[f.index,'signup_date'])).dt.days
    f['lapsed']=(~f.index.isin(set(sept))).astype(int)
    f=f.reset_index().rename(columns={'index':'customer_id'}).fillna(0)
    y=f.pop('lapsed').astype(int); ids=f.pop('customer_id'); cats=['fav_category','city','gender']; nums=[z for z in f.columns if z not in cats]
    X=f.copy(); Xtr,Xte,ytr,yte, idtr,idte=train_test_split(X,y,ids,test_size=.2,stratify=y,random_state=42)
    preproc=ColumnTransformer([('num',StandardScaler(),nums),('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),cats)])
    lr=Pipeline([('prep',preproc),('model',LogisticRegression(max_iter=1000))]); rf=Pipeline([('prep',preproc),('model',RandomForestClassifier(n_estimators=100,random_state=42))])
    gl=GridSearchCV(lr,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1',n_jobs=1).fit(Xtr,ytr)
    gr=GridSearchCV(rf,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=1).fit(Xtr,ytr)
    def metrics(model):
        pred=model.predict(Xte); return {'confusion_matrix':confusion_matrix(yte,pred).tolist(),'precision':precision_score(yte,pred,zero_division=0),'recall':recall_score(yte,pred,zero_division=0),'f1':f1_score(yte,pred,zero_division=0)}
    baseline_pred=np.zeros(len(yte),dtype=int)
    baseline={'accuracy':float(accuracy_score(yte,baseline_pred)),
              'f1_lapsed':float(f1_score(yte,baseline_pred,zero_division=0))}
    print('Lapse cohort',len(f),'lapsed',int(y.sum()),'train/test',len(ytr),len(yte),'baseline',baseline,
          'LR',gl.best_params_,gl.best_score_,metrics(gl),'RF',gr.best_params_,gr.best_score_,metrics(gr))
    best=gr if metrics(gr)['f1']>=metrics(gl)['f1'] else gl
    # persist a self-contained model pipeline
    joblib.dump(best,'outputs/lapse_model.joblib'); loaded=joblib.load('outputs/lapse_model.joblib'); assert np.array_equal(best.predict(Xte),loaded.predict(Xte))
    # class balance resampling on train only, best logistic model refit on balanced training rows
    pos=Xtr[ytr.eq(1)]; neg=Xtr[ytr.eq(0)]; yp=ytr
    if len(pos) and len(neg):
        pos_extra=pos.sample(n=max(0,len(neg)-len(pos)),replace=True,random_state=42); xb=pd.concat([Xtr, pos_extra]); yb=pd.concat([ytr,pd.Series([1]*len(pos_extra),index=pos_extra.index)])
        bal=Pipeline([('prep',preproc),('model',LogisticRegression(max_iter=1000,C=float(gl.best_params_['model__C'])))]).fit(xb,yb)
        balanced_metrics=metrics(bal); print('Balanced LR',balanced_metrics)
    else:
        balanced_metrics={}
    # Save model diagnostics so the README can report interpretable evidence.
    rf_best=gr.best_estimator_; lr_best=gl.best_estimator_
    rf_names=rf_best.named_steps['prep'].get_feature_names_out()
    rf_imp=rf_best.named_steps['model'].feature_importances_
    top_features=sorted([{'feature':str(n),'importance':float(v)} for n,v in zip(rf_names,rf_imp)],key=lambda a:a['importance'],reverse=True)[:5]
    lr_names=lr_best.named_steps['prep'].get_feature_names_out(); lr_coef=lr_best.named_steps['model'].coef_[0]
    coeffs=sorted([{'feature':str(n),'coefficient':float(v),'sign':'positive' if v>0 else 'negative' if v<0 else 'zero'} for n,v in zip(lr_names,lr_coef)],key=lambda a:abs(a['coefficient']),reverse=True)[:10]
    scaler_means=lr_best.named_steps['prep'].named_transformers_['num'].mean_
    freq_idx=nums.index('frequency')
    diagnostics={'cohort_size':int(len(f)),'lapsed_count':int(y.sum()),'train_size':int(len(ytr)),'test_size':int(len(yte)),
      'baseline':baseline,'logistic_regression':{'best_params':gl.best_params_,'best_cv_f1':float(gl.best_score_),'test':metrics(gl)},
      'random_forest':{'best_params':gr.best_params_,'best_cv_f1':float(gr.best_score_),'test':metrics(gr)},
      'selected_model':'random_forest' if best is gr else 'logistic_regression','balanced_logistic_test':balanced_metrics,
      'training_scaler_frequency_mean':float(scaler_means[freq_idx]),'full_cohort_frequency_mean':float(f['frequency'].mean()),
      'random_forest_top_features':top_features,'logistic_regression_largest_coefficients':coeffs}
    Path('outputs/model_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    # illustrative post-hoc Premium list based on whole-window RFM
    seg=rfm[['customer_id','is_premium']]; test_ids=pd.DataFrame({'customer_id':idte.values,'prob':best.predict_proba(Xte)[:,1],'actually_lapsed':yte.values})
    top=test_ids.merge(seg,on='customer_id'); top=top[top.is_premium].sort_values('prob',ascending=False).head(10)
    top.rename(columns={'prob':'lapse_probability'}).drop(columns=['is_premium']).to_csv('outputs/retention_list.csv',index=False)
    print('Premium retention list true lapsed',int(top.actually_lapsed.sum()))
    return rfm

def part_c(c):
    tickets=pd.read_csv('data/raw/tickets.csv',keep_default_na=False); seg=pd.read_csv('outputs/customer_segments.csv')
    patterns=[(re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'),'[EMAIL]'),(re.compile(r'\b[6-9]\d{9}\b'),'[PHONE]'),(re.compile(r'ORD-\d{6}'),'[ORDER_ID]')]
    clean=[]; counts=[0,0,0]
    for t in tickets.text.astype(str):
        s=t
        for i,(pat,repl) in enumerate(patterns):s,n=pat.subn(repl,s); counts[i]+=n
        clean.append(s)
        assert all(not pat.search(s) for pat,_ in patterns)
    zero='CLASSIFY. Categories: billing, delivery, technical, fraud. Return JSON only: {"category":"...","reason":"..."}.'
    few=zero+'\nEXAMPLES:\nTicket: unauthorized transaction on my card. Output: {"category":"fraud","reason":"unauthorized transaction"}\nTicket: app crashes on login. Output: {"category":"technical","reason":"app login crash"}'
    def run(prompt,budget=None):
        mock_llm.reset(); rows=[]; tokens=0; errors=0; eligible=0
        for (_,t),txt in zip(tickets.iterrows(),clean):
            row={'ticket_id':t.ticket_id,'customer_id':t.customer_id,'status':'','category':'','tokens_used':0,'true_category':t.true_category}
            if not txt.strip(): row['status']='skipped_empty'
            elif mock_llm.est_tokens(txt)>400: row['status']='flagged_long'
            elif budget is not None and tokens>=budget: row['status']='budget_exhausted'
            else:
                eligible+=1
                msg=[{'role':'system','content':prompt},{'role':'user','content':txt}]
                attempt=0
                while True:
                    try: resp=mock_llm.chat(msg); break
                    except mock_llm.TransientLLMError:
                        if attempt>=3: raise
                        wait=.05*(2**attempt)*random.uniform(.9,1.15); print(f"retry {t.ticket_id} retry={attempt} wait={wait:.4f}"); time.sleep(wait); attempt+=1; errors+=1
                used=resp.usage.total_tokens; tokens+=used; row['tokens_used']=used
                raw=resp.choices[0].message.content
                try:
                    obj=json.loads(raw); cat=obj.get('category')
                    if cat in ['billing','delivery','technical','fraud']:row.update(status='classified_json',category=cat)
                    else:row['status']='flagged_invalid_output'
                except Exception:
                    words=re.findall(r'[a-z]+',(raw or '').lower()); hits=[x for x in ['billing','delivery','technical','fraud'] if x in words]
                    if len(hits)==1:row.update(status='classified_keyword',category=hits[0])
                    else:row['status']='flagged_invalid_output'
            rows.append(row)
        df=pd.DataFrame(rows); classified=df[df.status.isin(['classified_json','classified_keyword'])]
        acc=(classified.category==classified.true_category).mean() if len(classified) else 0
        return df,tokens,acc,errors
    z,zt,za,ze=run(zero); f,ft,fa,fe=run(few)
    z.to_csv('outputs/triage_zero_shot.csv',index=False)
    print('Redaction counts',counts,sum(counts),'zero-shot',len(z[z.status.str.startswith('classified')]),za,zt,'few-shot',len(f[f.status.str.startswith('classified')]),fa,ft,'errors',ze,fe)
    zc=z[z.status.isin(['classified_json','classified_keyword'])]
    confusion=pd.crosstab(zc.true_category,zc.category).reindex(index=['billing','delivery','technical','fraud'],columns=['billing','delivery','technical','fraud'],fill_value=0)
    bd,_,_,_=run(zero,budget=6000); bd.to_csv('outputs/triage_budget_demo.csv',index=False)
    summary={'redaction_substitutions':{'email':counts[0],'phone':counts[1],'order_id':counts[2],'total':sum(counts)},
      'zero_shot':{'classified':int(len(zc)),'accuracy':float(za),'tokens':int(zt),'tokens_per_classified':float(zt/max(1,len(zc))),
                   'retry_errors_recovered':int(ze),'status_counts':{str(k):int(v) for k,v in z.status.value_counts().items()},
                   'confusion_matrix':confusion.to_dict()},
      'few_shot':{'classified':int(len(f[f.status.isin(['classified_json','classified_keyword'])])),'accuracy':float(fa),'tokens':int(ft),
                  'tokens_per_classified':float(ft/max(1,len(f[f.status.isin(['classified_json','classified_keyword'])]))),
                  'retry_errors_recovered':int(fe),'status_counts':{str(k):int(v) for k,v in f.status.value_counts().items()}},
      'budget_demo':{'budget_tokens':6000,'budget_exhausted':int((bd.status=='budget_exhausted').sum()),
                     'status_counts':{str(k):int(v) for k,v in bd.status.value_counts().items()}}}
    Path('outputs/triage_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    f,_,_,_=run(few,budget=100000)
    out=[]
    segmap=seg.set_index('customer_id').to_dict('index')
    for _,r in f.iterrows():
        info=segmap.get(r.customer_id,{}); premium=bool(info.get('is_premium',False)); name=info.get('segment_name','')
        if r.status in ['skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted']: route,priority='human_review','NORMAL'
        elif r.category=='fraud':route,priority='security_team','HIGH'
        elif premium:route,priority='premium_desk','HIGH'
        else:route,priority='support_queue','NORMAL'
        out.append({'ticket_id':r.ticket_id,'customer_id':r.customer_id,'status':r.status,'category':r.category,'segment_name':name,'priority':priority,'route':route,'tokens_used':r.tokens_used})
    pd.DataFrame(out).to_csv('outputs/triage_results.csv',index=False)

def part_d():
    docs={p.name:p.read_text(encoding='utf-8') for p in Path('data/policies').glob('*.txt')}
    def chunks(overlap=10):
        out=[]
        for source,text in docs.items():
            words=text.split(); step=40-overlap
            for start in range(0,len(words),step):
                part=words[start:start+40]
                if not part:break
                out.append({'source':source,'text':' '.join(part)})
                if start+40>=len(words):break
        return out
    def make_retriever(overlap=10):
        ch=chunks(overlap); vec=TfidfVectorizer(); mat=vec.fit_transform([x['text'] for x in ch])
        def retrieve(q,k=3):
            sims=cosine_similarity(vec.transform([q]),mat).ravel(); ix=np.argsort(-sims)[:k]
            return [{'source':ch[i]['source'],'score':float(sims[i]),'text':ch[i]['text']} for i in ix]
        return ch,retrieve
    ch,retrieve=make_retriever()
    def answer(q,ret=retrieve):
        rs=ret(q,3); context='\n'.join(f"[{x['source']}] {x['text']}" for x in rs)
        resp=mock_llm.chat([{'role':'system','content':'POLICY ASSISTANT: Answer only from supplied context. If absent, say I could not find that information in the policy documents.'},{'role':'user','content':f'CONTEXT:\n{context}\nQUESTION:\n{q}'}])
        return resp.choices[0].message.content,rs
    _RUNTIME.update(retrieve=retrieve,answer=answer)
    ev=pd.read_csv('data/raw/policy_eval.csv',keep_default_na=False); hits_doc=hits_fact=fallback=0
    for _,r in ev.iterrows():
        rs=retrieve(r.question,3); ans,_=answer(r.question)
        if r.expected_doc!='NONE':hits_doc+=int(r.expected_doc in [x['source'] for x in rs]); hits_fact+=int(r.expected_fact in ans)
        else:fallback+=int(ans.startswith('I could not find'))
    ch0,ret0=make_retriever(0); hd0=hf0=fb0=0; zero_overlap_failures=[]
    for _,r in ev.iterrows():
        rs=ret0(r.question,3); ans,_=answer(r.question,ret0)
        if r.expected_doc!='NONE':
            hd0+=int(r.expected_doc in [x['source'] for x in rs]); matched=int(r.expected_fact in ans); hf0+=matched
            if not matched:zero_overlap_failures.append({'question':r.question,'expected_fact':r.expected_fact,'answer':ans})
        else:fb0+=int(ans.startswith('I could not find'))
    rag_summary={'chunks':len(ch),'policy_eval_questions':int(len(ev)),'answerable_questions':int((ev.expected_doc!='NONE').sum()),
      'overlap_10':{'expected_doc_hits':int(hits_doc),'expected_fact_matches':int(hits_fact),'fallbacks':int(fallback)},
      'overlap_0':{'expected_doc_hits':int(hd0),'expected_fact_matches':int(hf0),'fallbacks':int(fb0)},
      'zero_overlap_failures':zero_overlap_failures}
    Path('outputs/rag_eval_summary.json').write_text(json.dumps(rag_summary,indent=2),encoding='utf-8')
    print('RAG chunks',len(ch),'eval overlap',hits_doc,hits_fact,fallback,'zero overlap',hd0,hf0,fb0)
    # Agent tools, bounded ReAct loop with durable checkpoints and resume support.
    ledger=[]
    def get_order(order_id):
        with sqlite3.connect('data/instamart.db') as db:
            row=db.execute('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?',(order_id,)).fetchone()
        return {'error':'order not found'} if row is None else dict(zip(['order_id','status','order_amount','delivery_minutes'],row))
    def search_policy(query):return {'results':retrieve(query,3)}
    def calculate_refund(order_amount,delivery_minutes):
        return {'refund_inr':round(float(order_amount),2) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0}
    def action(tool_name,order_id,refund_inr):
        row={'timestamp':time.time(),'tool':tool_name,'order_id':order_id,'refund_inr':float(refund_inr)}
        ledger.append(row); dump_jsonl('outputs/refund_ledger.jsonl',ledger)
        return {'refund_inr':float(refund_inr)}
    def issue_refund(order_id,refund_inr):return action('issue_refund',order_id,refund_inr)
    def escalate_to_human(order_id,refund_inr):return action('escalate_to_human',order_id,refund_inr)
    toolmap={'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':issue_refund,'escalate_to_human':escalate_to_human}
    spec=[
      {'name':'get_order','description':'Look up an order and its delivery facts.','parameters':{'type':'object','properties':{'order_id':{'type':'string'}},'required':['order_id']}},
      {'name':'search_policy','description':'Retrieve the most relevant policy chunks.','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}},
      {'name':'calculate_refund','description':'Calculate a full refund only for deliveries taking more than 45 minutes.','parameters':{'type':'object','properties':{'order_amount':{'type':'number'},'delivery_minutes':{'type':['number','null']}},'required':['order_amount','delivery_minutes']}},
      {'name':'issue_refund','description':'Record an approved automatic refund of INR 1000 or less.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}},
      {'name':'escalate_to_human','description':'Record escalation for a refund above INR 1000.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}}
    ]
    TOOL_MAP=toolmap
    TOOLS_SPEC=spec
    class SimulatedCrash(RuntimeError):pass
    def _checkpoint(run_id,scenario_id,msgs,tokens,calls,steps):
        Path('state').mkdir(exist_ok=True)
        payload={'run_id':run_id,'scenario_id':scenario_id,'messages':msgs,'tokens_used':tokens,
                 'step_number':len(steps),'tools_executed':[x['tool'] for x in steps],
                 'steps':steps,'llm_calls':calls}
        Path(f'state/{run_id}.json').write_text(json.dumps(payload,default=str,ensure_ascii=False,indent=2),encoding='utf-8')
    def _continue_agent(user_message,run_id,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,
                        resume_state=None,crash_after_tool_count=None):
        if resume_state is None:
            msgs=[{'role':'system','content':'AGENT: use tools to decide refund safely.'},{'role':'user','content':user_message}]
            steps=[]; tokens=0; calls=0
        else:
            msgs=resume_state['messages']; steps=resume_state.get('steps',[])
            tokens=int(resume_state['tokens_used']); calls=int(resume_state.get('llm_calls',0))
        start=time.monotonic()
        while calls<max_iterations:
            if time.monotonic()-start>=time_limit_seconds:
                return {'status':'FALLBACK_TIMEOUT','final_answer':'FALLBACK_TIMEOUT: hand off to a human with the steps completed.','steps':steps,'llm_calls':calls,'tokens_used':tokens}
            resp=mock_llm.chat(msgs,tools=TOOLS_SPEC); calls+=1; tokens+=resp.usage.total_tokens
            if tokens>token_limit:
                return {'status':'FALLBACK_TOKEN_LIMIT','final_answer':'FALLBACK_TOKEN_LIMIT: hand off to a human; token limit reached.','steps':steps,'llm_calls':calls,'tokens_used':tokens}
            m=resp.choices[0].message
            if not m.tool_calls:
                final=m.content or ''
                return {'status':final.split(':',1)[0],'final_answer':final,'steps':steps,'llm_calls':calls,'tokens_used':tokens}
            tc=m.tool_calls[0]; args=tc.arguments if isinstance(tc.arguments,dict) else json.loads(tc.arguments)
            msgs.append({'role':'assistant','content':None,'tool_calls':[{'id':tc.id,'name':tc.name,'arguments':args}]})
            obs=TOOL_MAP[tc.name](**args); steps.append({'tool':tc.name,'arguments':args,'observation':obs})
            msgs.append({'role':'tool','tool_call_id':tc.id,'name':tc.name,'content':json.dumps(obs)})
            # Persist only after the tool observation is in the transcript. A resumed
            # run starts after this observation and never replays its side effect.
            _checkpoint(run_id,scenario_id,msgs,tokens,calls,steps)
            if crash_after_tool_count is not None and len(steps)==crash_after_tool_count:
                raise SimulatedCrash(f'simulated crash after tool {len(steps)}')
        return {'status':'FALLBACK_MAX_ITERATIONS','final_answer':'FALLBACK_MAX_ITERATIONS: hand off to a human with the steps completed.','steps':steps,'llm_calls':calls,'tokens_used':tokens}
    def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,crash_after_tool_count=None):
        return _continue_agent(user_message,scenario_id,scenario_id,max_iterations,token_limit,time_limit_seconds,
                               resume_state=None,crash_after_tool_count=crash_after_tool_count)
    def resume_agent(run_id,max_iterations=6,token_limit=6000,time_limit_seconds=10):
        path=Path(f'state/{run_id}.json')
        if not path.exists():raise FileNotFoundError(f'No checkpoint found for run {run_id}')
        saved=json.loads(path.read_text(encoding='utf-8'))
        return _continue_agent('',run_id,saved['scenario_id'],max_iterations,token_limit,time_limit_seconds,resume_state=saved)
    _RUNTIME.update(run_agent=run_agent,resume_agent=resume_agent)
    globals()['TOOLS_SPEC']=TOOLS_SPEC; globals()['TOOL_MAP']=TOOL_MAP
    sc=pd.read_csv('data/raw/agent_scenarios.csv',keep_default_na=False); runs=[]
    for _,r in sc.iterrows():
        mock_llm.reset(); result=run_agent(r.message,r.scenario_id)
        runs.append({'scenario_id':r.scenario_id,**result})
    dump_jsonl('outputs/agent_runs.jsonl',runs); dump_jsonl('outputs/refund_ledger.jsonl',ledger)
    print('Agent scenarios',[(r['scenario_id'],r['status'],r['llm_calls']) for r in runs])
    print('Stop-safe demos',run_agent(sc.iloc[0].message,'demo_token',token_limit=200)['status'],run_agent(sc.iloc[0].message,'demo_timeout',time_limit_seconds=0)['status'])
    # Simulate a crash after the second observation of S1, then resume from its
    # saved checkpoint. The already completed tools must not be executed twice.
    resume_run_id='S1_resume_demo'; resume_message=sc.loc[sc.scenario_id.eq('S1'),'message'].iloc[0]
    Path(f'state/{resume_run_id}.json').unlink(missing_ok=True); mock_llm.reset()
    try:
        run_agent(resume_message,resume_run_id,crash_after_tool_count=2)
    except SimulatedCrash:
        pass
    resumed=resume_agent(resume_run_id)
    executed=[step['tool'] for step in resumed['steps']]
    assert executed==['get_order','search_policy','calculate_refund','issue_refund'],executed
    assert resumed['status']=='REFUND' and abs(resumed['steps'][-1]['observation']['refund_inr']-128.25)<0.01
    Path('outputs/resume_demo.json').write_text(json.dumps({'scenario_id':'S1','status':resumed['status'],
       'final_answer':resumed['final_answer'],'tools_executed':executed,'llm_calls':resumed['llm_calls'],
       'tokens_used':resumed['tokens_used']},indent=2),encoding='utf-8')
    print('Crash/resume demo',resumed['status'],executed)

def retrieve(query,k=3):
    if 'retrieve' not in _RUNTIME:raise RuntimeError('Run part_d() to initialise the policy retriever first.')
    return _RUNTIME['retrieve'](query,k)

def answer(question):
    if 'answer' not in _RUNTIME:raise RuntimeError('Run part_d() to initialise the policy assistant first.')
    return _RUNTIME['answer'](question)

def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10):
    if 'run_agent' not in _RUNTIME:raise RuntimeError('Run part_d() to initialise the refund agent first.')
    return _RUNTIME['run_agent'](user_message,scenario_id,max_iterations,token_limit,time_limit_seconds)

def resume_agent(run_id,max_iterations=6,token_limit=6000,time_limit_seconds=10):
    if 'resume_agent' not in _RUNTIME:raise RuntimeError('Run part_d() to initialise the refund agent first.')
    return _RUNTIME['resume_agent'](run_id,max_iterations,token_limit,time_limit_seconds)

if __name__=='__main__':
    if not Path('data/raw/customers.csv').exists():
        import generate_data
    c,p,o=part_a(); part_b(c,p,o); part_c(c); part_d()
    print('Pipeline finished')
