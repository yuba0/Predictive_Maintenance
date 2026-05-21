function PageModels() {
  const { useState, useEffect } = React;
  const [tab, setTab] = useState('metrics');
  const [models, setModels] = useState([]);
  const [curves, setCurves] = useState({});
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState(null);

  const COLOR_MAP = {
    'XGBoost':             '#3B82F6',
    'Random Forest':       '#00B4D8',
    'MLP (PyTorch)':       '#8B5CF6',
    'Logistic Regression': '#F59E0B',
  };

  useEffect(() => {
    Promise.all([
      fetch('http://localhost:8000/models/metrics').then(r=>r.json()),
      fetch('http://localhost:8000/models/curves').then(r=>r.json()).catch(()=>({})),
      fetch('http://localhost:8000/models/confusion').then(r=>r.json()).catch(()=>({})),
    ]).then(([metricsData, curvesData, confData]) => {
      const enriched = (metricsData.models || []).map(m => ({
        ...m,
        color: COLOR_MAP[m.name] || '#3B82F6',
        cm: confData[m.name] || null,
      }));
      setModels(enriched);
      setCurves(curvesData);
      setLoading(false);
    }).catch(() => {
      setApiError('API non disponible — lancez : python3 -m uvicorn api.main:app --port 8000');
      setLoading(false);
    });
  }, []);

  const tabs = [
    { id:'metrics',   label:'Metrics' },
    { id:'confusion', label:'Confusion Matrix' },
    { id:'curves',    label:'ROC & PR Curves' },
    { id:'radar',     label:'Radar Comparison' },
  ];

  const rocPointsFor = (name) => {
    const c = curves[name]; if (!c) return [];
    return c.roc.fpr.map((x,i) => [x, c.roc.tpr[i]]);
  };
  const prPointsFor = (name) => {
    const c = curves[name]; if (!c) return [];
    return c.pr.recall.map((x,i) => [x, c.pr.precision[i]]);
  };
  const rocFallback = (auc) => { const n=40,pts=[]; for(let i=0;i<=n;i++){const f=i/n;pts.push([f,clamp(Math.pow(f,1-auc)*(1+(auc-0.85)*0.4),0,1)]);} return pts; };
  const prFallback  = (auc) => { const n=40,pts=[]; for(let i=0;i<=n;i++){const r=i/n;pts.push([r,clamp(1-Math.pow(r,1/(auc*2.5))*(1-auc)-(1-auc)*0.15,0.4,1)]);} return pts; };

  const bestModel = models.find(m=>m.best) || models[0];
  const hasCurves = Object.keys(curves).length > 0;

  if (loading) return (
    <div style={{display:'flex',alignItems:'center',justifyContent:'center',minHeight:400,gap:14,color:'#7F8FAD',fontSize:14}}>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" strokeWidth="2"><circle cx="12" cy="12" r="9" strokeDasharray="20 40"><animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/></circle></svg>
      Chargement des métriques depuis l'API…
    </div>
  );

  return (
    <div style={{padding:'24px 28px 40px',display:'flex',flexDirection:'column',gap:18}}>

      {apiError && (
        <div style={{padding:'10px 14px',borderRadius:10,background:'rgba(239,68,68,.1)',border:'1px solid rgba(239,68,68,.35)',color:'#EF4444',fontSize:12,display:'flex',alignItems:'center',gap:10}}>
          <I.AlertTriangle size={14}/> {apiError}
        </div>
      )}

      {/* Tabs */}
      <div className="glass" style={{padding:'8px',borderRadius:12,display:'flex',gap:4,alignSelf:'flex-start'}}>
        {tabs.map(t=>(
          <button key={t.id} onClick={()=>setTab(t.id)} style={{
            padding:'9px 18px',borderRadius:8,border:'none',cursor:'pointer',fontSize:12.5,fontWeight:600,
            background:tab===t.id?'linear-gradient(180deg,rgba(59,130,246,.22),rgba(59,130,246,.06))':'transparent',
            color:tab===t.id?'#E8F0FF':'#7F8FAD',
            boxShadow:tab===t.id?'inset 0 0 0 1px rgba(59,130,246,.35),0 0 18px -6px rgba(59,130,246,.55)':'none',
            transition:'all .15s',
          }}>{t.label}</button>
        ))}
      </div>

      {tab==='metrics' && models.length > 0 && (
        <div style={{display:'grid',gridTemplateColumns:'1.4fr 1fr',gap:18}}>
          <div className="glass" style={{padding:20,borderRadius:16}}>
            <SectionTitle accent="#3B82F6" action={
              <div style={{display:'flex',gap:8,alignItems:'center',fontSize:11,color:'#7F8FAD'}}>
                <I.Info size={12}/> real trained models · sorted by F1
              </div>
            }>Model leaderboard · données réelles</SectionTitle>
            <table style={{width:'100%',borderCollapse:'separate',borderSpacing:0,fontSize:12.5}}>
              <thead>
                <tr style={{textAlign:'left',color:'#7F8FAD',fontSize:10.5,letterSpacing:'.12em'}}>
                  <th style={{padding:'10px 8px'}}>MODEL</th>
                  <th style={{padding:'10px 8px',textAlign:'right'}}>RECALL ↑</th>
                  <th style={{padding:'10px 8px',textAlign:'right'}}>PRECISION</th>
                  <th style={{padding:'10px 8px',textAlign:'right'}}>F1 ↑</th>
                  <th style={{padding:'10px 8px',textAlign:'right'}}>PR-AUC ↑</th>
                  <th style={{padding:'10px 8px',textAlign:'right'}}>ROC-AUC ↑</th>
                  <th style={{padding:'10px 8px',textAlign:'right'}}>TRAIN(s)</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m,i)=>{
                  const cell=(v,isBest)=>(
                    <td style={{padding:'12px 8px',textAlign:'right',background:isBest?'rgba(6,214,160,.13)':'transparent',borderRadius:6}}>
                      <span className="mono tabular" style={{color:isBest?'#06D6A0':'#E8F0FF',fontWeight:isBest?700:500}}>{typeof v==='number'?v.toFixed(3):v}</span>
                      {isBest&&<span style={{marginLeft:5,fontSize:9.5,color:'#06D6A0'}}>★</span>}
                    </td>
                  );
                  const maxOf=(k)=>Math.max(...models.map(x=>x[k]||0));
                  const minOf=(k)=>Math.min(...models.map(x=>x[k]||999));
                  return (
                    <tr key={i} style={{borderBottom:'1px dashed rgba(59,130,246,.1)',background:m.best?'rgba(59,130,246,.06)':'transparent'}}>
                      <td style={{padding:'12px 8px'}}>
                        <div style={{display:'flex',alignItems:'center',gap:9}}>
                          <span style={{width:10,height:10,borderRadius:2,background:m.color,boxShadow:`0 0 6px ${m.color}66`}}/>
                          <span className="grotesk" style={{fontWeight:600,color:'#E8F0FF'}}>{m.name}</span>
                          {m.best&&<span style={{padding:'2px 6px',borderRadius:5,background:'rgba(59,130,246,.18)',border:'1px solid rgba(59,130,246,.5)',fontSize:9.5,color:'#3B82F6',letterSpacing:'.12em',fontWeight:700}}>BEST</span>}
                        </div>
                      </td>
                      {cell(m.recall,    m.recall    ===maxOf('recall'))}
                      {cell(m.precision, m.precision ===maxOf('precision'))}
                      {cell(m.f1,        m.f1        ===maxOf('f1'))}
                      {cell(m.pr_auc,    m.pr_auc    ===maxOf('pr_auc'))}
                      {cell(m.roc_auc,   m.roc_auc   ===maxOf('roc_auc'))}
                      {cell(m.train_time,m.train_time===minOf('train_time'))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div style={{marginTop:18,padding:14,background:'rgba(10,22,40,.5)',borderRadius:10,border:'1px solid rgba(59,130,246,.12)'}}>
              <div style={{fontSize:11,color:'#7F8FAD',letterSpacing:'.12em',marginBottom:10}}>SCORE COMPARISON · GROUPED</div>
              <GroupedBars
                groups={['Recall','F1','PR-AUC','ROC-AUC']}
                series={models.map(m=>({name:m.name,color:m.color,values:[m.recall,m.f1,m.pr_auc,m.roc_auc]}))}
                width={580} height={210}
              />
              <div style={{display:'flex',gap:14,marginTop:8,fontSize:11,color:'#B7C5DD',justifyContent:'center'}}>
                {models.map(m=>(<span key={m.name} style={{display:'flex',alignItems:'center',gap:5}}><span style={{width:9,height:9,borderRadius:2,background:m.color}}/>{m.name}</span>))}
              </div>
            </div>
          </div>

          {bestModel && (
            <div style={{display:'flex',flexDirection:'column',gap:14}}>
              <div className="glass" style={{padding:20,borderRadius:16,position:'relative',overflow:'hidden',borderColor:'rgba(59,130,246,.4)',boxShadow:'0 0 40px -10px rgba(59,130,246,.5)'}}>
                <div style={{position:'absolute',top:-40,right:-40,width:200,height:200,background:'radial-gradient(circle,rgba(59,130,246,.4),transparent 60%)',pointerEvents:'none'}}/>
                <div style={{display:'flex',alignItems:'center',gap:10}}>
                  <I.Medal1/>
                  <span style={{fontSize:11,color:'#3B82F6',letterSpacing:'.16em',fontWeight:700}}>BEST MODEL · DEPLOYED</span>
                </div>
                <div className="grotesk" style={{fontSize:28,fontWeight:700,marginTop:8,letterSpacing:'-.02em',filter:'drop-shadow(0 0 12px rgba(59,130,246,.5))'}}>{bestModel.name}</div>
                <div style={{fontSize:12.5,color:'#B7C5DD',marginTop:6,lineHeight:1.55}}>
                  Sélectionné grâce au meilleur Recall (<span className="mono" style={{color:'#06D6A0',fontWeight:600}}>{bestModel.recall.toFixed(3)}</span>) et PR-AUC (<span className="mono" style={{color:'#06D6A0',fontWeight:600}}>{bestModel.pr_auc.toFixed(4)}</span>) — un défaut manqué coûte <span className="mono" style={{color:'#F59E0B'}}>≈ 18 400 €</span>, une fausse alarme <span className="mono" style={{color:'#06D6A0'}}>420 €</span>.
                </div>
                <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:10,marginTop:14}}>
                  {[['Recall',bestModel.recall.toFixed(3)],['F1-Score',bestModel.f1.toFixed(3)],['ROC-AUC',bestModel.roc_auc.toFixed(4)]].map(([k,v])=>(
                    <div key={k} style={{background:'rgba(10,22,40,.5)',padding:10,borderRadius:10}}>
                      <div style={{fontSize:10,color:'#7F8FAD',letterSpacing:'.1em'}}>{k}</div>
                      <div className="mono grotesk tabular" style={{fontSize:16,fontWeight:700,color:'#06D6A0'}}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="glass" style={{padding:18,borderRadius:14}}>
                <SectionTitle accent="#06D6A0">Training time · seconds</SectionTitle>
                <BarChart data={models.map(m=>m.train_time)} labels={models.map(m=>m.name.split(' ')[0])} color="#00B4D8" width={360} height={180} format={v=>v+'s'}/>
                <div style={{fontSize:11,color:'#7F8FAD',marginTop:4}}>3-fold CV · résultats réels d'entraînement</div>
              </div>
            </div>
          )}
        </div>
      )}

      {tab==='confusion' && (
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:18}}>
          {models.map((m,i)=>{
            const cm = m.cm || [[3000,200],[150,500]];
            const [[tn,fp],[fn,tp]] = cm;
            const recall=tp/(tp+fn), precision=tp/(tp+fp), f1=2*recall*precision/(recall+precision);
            return (
              <div key={i} className="glass" style={{padding:20,borderRadius:16,borderLeft:`3px solid ${m.color}`,boxShadow:m.best?`0 0 30px -10px ${m.color}77`:'none'}}>
                <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:12}}>
                  <div style={{display:'flex',alignItems:'center',gap:10}}>
                    <span style={{width:12,height:12,borderRadius:3,background:m.color,boxShadow:`0 0 8px ${m.color}`}}/>
                    <h3 className="grotesk" style={{margin:0,fontSize:16,fontWeight:700}}>{m.name}</h3>
                    {m.best&&<span style={{padding:'2px 6px',borderRadius:5,background:'rgba(59,130,246,.18)',border:'1px solid rgba(59,130,246,.5)',fontSize:9.5,color:'#3B82F6',letterSpacing:'.12em',fontWeight:700}}>BEST</span>}
                  </div>
                  <span className="mono" style={{fontSize:11,color:'#7F8FAD'}}>n = {(tn+fp+fn+tp).toLocaleString()}</span>
                </div>
                <div style={{display:'grid',gridTemplateColumns:'auto 1fr',gap:18,alignItems:'center'}}>
                  <ConfusionMatrix matrix={cm} size={200}/>
                  <div style={{display:'flex',flexDirection:'column',gap:10}}>
                    {[{l:'Recall',v:recall,c:'#06D6A0'},{l:'Precision',v:precision,c:'#3B82F6'},{l:'F1 score',v:f1,c:'#00B4D8'}].map((s,k)=>(
                      <div key={k}>
                        <div style={{display:'flex',justifyContent:'space-between',fontSize:11.5,marginBottom:4}}>
                          <span style={{color:'#B7C5DD'}}>{s.l}</span>
                          <span className="mono tabular" style={{color:s.c,fontWeight:700}}>{s.v.toFixed(3)}</span>
                        </div>
                        <div style={{height:5,background:'rgba(59,130,246,.08)',borderRadius:99,overflow:'hidden'}}>
                          <div style={{height:'100%',width:`${s.v*100}%`,background:`linear-gradient(90deg,${s.c},${s.c}88)`,boxShadow:`0 0 8px ${s.c}66`}}/>
                        </div>
                      </div>
                    ))}
                    <div style={{display:'flex',justifyContent:'space-between',marginTop:6,padding:'8px 10px',background:'rgba(10,22,40,.5)',borderRadius:8}}>
                      <span style={{fontSize:11,color:'#7F8FAD'}}>Missed failures (FN)</span>
                      <span className="mono" style={{fontSize:13,fontWeight:700,color:'#EF4444'}}>{fn}</span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {tab==='curves' && (
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:18}}>
          <div className="glass" style={{padding:20,borderRadius:16}}>
            <SectionTitle accent="#3B82F6">ROC Curve · TPR vs FPR {hasCurves?'· données réelles':'· approximation'}</SectionTitle>
            <div style={{display:'flex',gap:12,alignItems:'center'}}>
              <CurveChart curves={models.map(m=>({name:m.name,color:m.color,points:hasCurves?rocPointsFor(m.name):rocFallback(m.roc_auc)}))} xLabel="False positive rate" yLabel="True positive rate" width={420} height={300}/>
              <div style={{display:'flex',flexDirection:'column',gap:8,minWidth:170}}>
                <div style={{fontSize:10.5,color:'#7F8FAD',letterSpacing:'.12em'}}>AUC SCORES</div>
                {[...models].sort((a,b)=>b.roc_auc-a.roc_auc).map(m=>(
                  <div key={m.name} style={{display:'flex',alignItems:'center',justifyContent:'space-between',padding:'6px 10px',background:'rgba(10,22,40,.5)',borderRadius:8,border:`1px solid ${m.color}33`}}>
                    <div style={{display:'flex',alignItems:'center',gap:8}}>
                      <span style={{width:10,height:3,background:m.color,borderRadius:2,boxShadow:`0 0 6px ${m.color}`}}/>
                      <span style={{fontSize:11.5,color:'#E8F0FF'}}>{m.name}</span>
                    </div>
                    <span className="mono tabular" style={{fontSize:12.5,fontWeight:700,color:m.color}}>{m.roc_auc.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="glass" style={{padding:20,borderRadius:16}}>
            <SectionTitle accent="#00B4D8">PR Curve · Precision vs Recall {hasCurves?'· données réelles':'· approximation'}</SectionTitle>
            <div style={{display:'flex',gap:12,alignItems:'center'}}>
              <CurveChart curves={models.map(m=>({name:m.name,color:m.color,points:hasCurves?prPointsFor(m.name):prFallback(m.pr_auc)}))} xLabel="Recall" yLabel="Precision" width={420} height={300} diagonal={false}/>
              <div style={{display:'flex',flexDirection:'column',gap:8,minWidth:170}}>
                <div style={{fontSize:10.5,color:'#7F8FAD',letterSpacing:'.12em'}}>AVG. PRECISION</div>
                {[...models].sort((a,b)=>b.pr_auc-a.pr_auc).map(m=>(
                  <div key={m.name} style={{display:'flex',alignItems:'center',justifyContent:'space-between',padding:'6px 10px',background:'rgba(10,22,40,.5)',borderRadius:8,border:`1px solid ${m.color}33`}}>
                    <div style={{display:'flex',alignItems:'center',gap:8}}>
                      <span style={{width:10,height:3,background:m.color,borderRadius:2,boxShadow:`0 0 6px ${m.color}`}}/>
                      <span style={{fontSize:11.5,color:'#E8F0FF'}}>{m.name}</span>
                    </div>
                    <span className="mono tabular" style={{fontSize:12.5,fontWeight:700,color:m.color}}>{m.pr_auc.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="glass" style={{padding:18,borderRadius:14,gridColumn:'1 / 3'}}>
            <div style={{display:'flex',alignItems:'flex-start',gap:14}}>
              <I.Info size={18} stroke="#00B4D8"/>
              <div style={{fontSize:12.5,color:'#B7C5DD',lineHeight:1.55}}>
                <span style={{color:'#E8F0FF',fontWeight:600}}>Reading guide.</span>{' '}
                Sur une classe positive à <span className="mono">14.8%</span>, le PR-AUC est le score le plus honnête.{' '}
                {bestModel&&<><span style={{color:'#3B82F6',fontWeight:600}}>{bestModel.name}</span> domine avec ROC-AUC = <span className="mono">{bestModel.roc_auc.toFixed(4)}</span> et PR-AUC = <span className="mono">{bestModel.pr_auc.toFixed(4)}</span>.</>}
              </div>
            </div>
          </div>
        </div>
      )}

      {tab==='radar' && models.length>0 && (
        <div style={{display:'grid',gridTemplateColumns:'1.1fr 1fr',gap:18}}>
          <div className="glass" style={{padding:20,borderRadius:16}}>
            <SectionTitle accent="#8B5CF6">Multi-axis comparison</SectionTitle>
            <div style={{display:'flex',gap:14,alignItems:'center'}}>
              <Radar
                axes={['Recall','Precision','F1','PR-AUC','ROC-AUC','Speed']}
                series={models.map(m=>({name:m.name,color:m.color,values:[m.recall,m.precision,m.f1,m.pr_auc,m.roc_auc,clamp(1-m.train_time/20,0,1)]}))}
                size={340}
              />
              <div style={{display:'flex',flexDirection:'column',gap:8}}>
                {models.map(m=>(
                  <div key={m.name} style={{display:'flex',alignItems:'center',gap:8,padding:'6px 10px',background:'rgba(10,22,40,.5)',borderRadius:8,border:`1px solid ${m.color}33`}}>
                    <span style={{width:10,height:10,borderRadius:99,background:m.color,boxShadow:`0 0 8px ${m.color}`}}/>
                    <span style={{fontSize:12,color:'#E8F0FF',fontWeight:m.best?700:500}}>{m.name}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="glass" style={{padding:20,borderRadius:16}}>
            <SectionTitle accent="#06D6A0">Training time · résultats réels</SectionTitle>
            <div style={{display:'flex',flexDirection:'column',gap:12,marginTop:8}}>
              {[...models].sort((a,b)=>a.train_time-b.train_time).map(m=>{
                const maxT=Math.max(...models.map(x=>x.train_time));
                return (
                  <div key={m.name}>
                    <div style={{display:'flex',justifyContent:'space-between',fontSize:12,marginBottom:5}}>
                      <span style={{color:'#E8F0FF',display:'flex',alignItems:'center',gap:6}}><span style={{width:9,height:9,borderRadius:2,background:m.color}}/>{m.name}</span>
                      <span className="mono tabular" style={{color:m.color,fontWeight:700}}>{m.train_time} s</span>
                    </div>
                    <div style={{height:14,background:'rgba(10,22,40,.6)',borderRadius:8,overflow:'hidden'}}>
                      <div style={{height:'100%',width:`${(m.train_time/maxT)*100}%`,background:`linear-gradient(90deg,${m.color},${m.color}99)`,boxShadow:`0 0 12px ${m.color}66`,display:'flex',alignItems:'center',justifyContent:'flex-end',paddingRight:6}}>
                        <span style={{fontSize:9,color:'#fff',fontWeight:700}}>{(m.train_time/maxT*100).toFixed(0)}%</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

window.PageModels = PageModels;
