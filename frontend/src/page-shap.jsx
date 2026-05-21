function PageShap() {
  const { useState, useEffect } = React;
  const [features, setFeatures] = useState([]);
  const [loading, setLoading] = useState(true);
  const [modelName, setModelName] = useState('');
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch('http://localhost:8000/models/shap').then(r=>r.json()),
      fetch('http://localhost:8000/health').then(r=>r.json()).catch(()=>({})),
    ]).then(([shapData, healthData]) => {
      const feats = (shapData.features || []).map(f => ({
        label: f.feature,
        value: f.value,
      })).sort((a,b)=>b.value-a.value);
      setFeatures(feats);
      setModelName(healthData.model || 'XGBoost');
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  const totalShap = features.reduce((s,f)=>s+f.value, 0);

  const medals = [<I.Medal1 key="1"/>, <I.Medal2 key="2"/>, <I.Medal3 key="3"/>,
    <span key="4" style={{fontSize:16,color:'#7F8FAD',fontWeight:700}}>4</span>,
    <span key="5" style={{fontSize:16,color:'#7F8FAD',fontWeight:700}}>5</span>];

  return (
    <div style={{padding:'24px 28px 40px',display:'flex',flexDirection:'column',gap:18}}>
      <div className="glass" style={{padding:20,borderRadius:16,position:'relative',overflow:'hidden'}}>
        <div style={{position:'absolute',top:-30,right:-30,width:240,height:240,background:'radial-gradient(circle,rgba(139,92,246,.35),transparent 60%)',pointerEvents:'none'}}/>
        <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',gap:18}}>
          <div>
            <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:6}}>
              <div style={{padding:'3px 10px',background:'rgba(139,92,246,.16)',border:'1px solid rgba(139,92,246,.45)',borderRadius:99,color:'#A78BFA',fontSize:10.5,fontWeight:700,letterSpacing:'.14em'}}>MODEL {modelName || '…'}</div>
              <div style={{padding:'3px 10px',background:'rgba(6,214,160,.12)',border:'1px solid rgba(6,214,160,.4)',borderRadius:99,color:'#06D6A0',fontSize:10.5,fontWeight:700,letterSpacing:'.14em'}}>SHAP TreeExplainer</div>
            </div>
            <h2 className="grotesk" style={{margin:0,fontSize:24,fontWeight:700,letterSpacing:'-.02em'}}>What drives the model failure predictions?</h2>
            <div style={{fontSize:13,color:'#B7C5DD',marginTop:6,maxWidth:680,lineHeight:1.55}}>
              SHAP (SHapley Additive exPlanations) décompose chaque prédiction en la contribution de chaque feature.
              Valeurs globales <span className="mono" style={{color:'#00B4D8'}}>mean |SHAP|</span> calculées sur l'ensemble de test complet.
            </div>
          </div>
          <div style={{display:'flex',flexDirection:'column',gap:8,minWidth:200}}>
            <div style={{padding:'10px 12px',background:'rgba(10,22,40,.5)',borderRadius:10,border:'1px solid rgba(59,130,246,.18)'}}>
              <div style={{fontSize:10,color:'#7F8FAD',letterSpacing:'.12em'}}>FEATURES ANALYZED</div>
              <div className="grotesk tabular" style={{fontSize:24,fontWeight:700}}>{loading?'…':features.length}</div>
            </div>
            <div style={{padding:'10px 12px',background:'rgba(10,22,40,.5)',borderRadius:10,border:'1px solid rgba(59,130,246,.18)'}}>
              <div style={{fontSize:10,color:'#7F8FAD',letterSpacing:'.12em'}}>TOP FEATURE SHAP</div>
              <div className="grotesk tabular" style={{fontSize:24,fontWeight:700,color:'#06D6A0'}}>{loading?'…':(features[0]?.value??0).toFixed(3)}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Top 5 */}
      {features.length > 0 && (
        <div>
          <SectionTitle accent="#8B5CF6">Top 5 most influential features · données réelles</SectionTitle>
          <div style={{display:'grid',gridTemplateColumns:'repeat(5,1fr)',gap:12}}>
            {features.slice(0,5).map((f,i)=>{
              const max = features[0].value;
              return (
                <div key={i} className="glass" style={{padding:14,borderRadius:14,position:'relative',overflow:'hidden',boxShadow:i<3?`0 0 24px -10px ${i===0?'#F59E0B66':i===1?'#B7C5DD66':'#CD7F3266'}`:'none'}}>
                  <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:8}}>
                    <span style={{fontSize:10,color:'#7F8FAD',letterSpacing:'.14em'}}>RANK</span>
                    {medals[i]}
                  </div>
                  <div className="mono" style={{fontSize:12,color:'#E8F0FF',fontWeight:600,minHeight:32}}>{f.label}</div>
                  <div className="grotesk tabular" style={{fontSize:22,fontWeight:700,color:'#3B82F6',marginTop:6,filter:'drop-shadow(0 0 8px rgba(59,130,246,.5))'}}>{f.value.toFixed(3)}</div>
                  <div style={{height:5,background:'rgba(59,130,246,.08)',borderRadius:99,marginTop:8,overflow:'hidden'}}>
                    <div style={{height:'100%',width:`${(f.value/max)*100}%`,background:'linear-gradient(90deg,#3B82F6,#00B4D8)',boxShadow:'0 0 8px rgba(59,130,246,.6)'}}/>
                  </div>
                  <div style={{fontSize:10,color:'#7F8FAD',marginTop:4}}>{((f.value/totalShap)*100).toFixed(1)}% of total</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {loading && (
        <div style={{display:'flex',alignItems:'center',justifyContent:'center',minHeight:200,gap:14,color:'#7F8FAD',fontSize:14}}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" strokeWidth="2"><circle cx="12" cy="12" r="9" strokeDasharray="20 40"><animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/></circle></svg>
          Chargement des valeurs SHAP…
        </div>
      )}

      {features.length > 0 && (
        <div style={{display:'grid',gridTemplateColumns:'1.2fr 1fr',gap:18}}>
          <div className="glass" style={{padding:20,borderRadius:16}}>
            <SectionTitle accent="#3B82F6" action={<span className="mono" style={{fontSize:10.5,color:'#7F8FAD'}}>mean |SHAP value| - sorted · API réelle</span>}>Global feature importance</SectionTitle>
            <HBarChart data={features} width={620} height={Math.max(300, features.length * 32)}/>
          </div>

          <div style={{display:'flex',flexDirection:'column',gap:14}}>
            <div className="glass" style={{padding:18,borderRadius:14,borderLeft:'3px solid #00B4D8',boxShadow:'inset 4px 0 16px -8px rgba(0,180,216,.4)'}}>
              <div style={{display:'flex',alignItems:'flex-start',gap:12}}>
                <div style={{width:36,height:36,borderRadius:10,background:'rgba(0,180,216,.15)',border:'1px solid rgba(0,180,216,.4)',display:'grid',placeItems:'center',color:'#00B4D8',flexShrink:0}}>
                  <I.Info size={18}/>
                </div>
                <div>
                  <div style={{fontSize:10.5,color:'#7F8FAD',letterSpacing:'.14em',marginBottom:4}}>NATURAL LANGUAGE EXPLANATION</div>
                  <div style={{fontSize:13,color:'#E8F0FF',lineHeight:1.6}}>
                    Le modèle place <span style={{color:'#3B82F6',fontWeight:600}}>{features[0]?.label}</span> en tête de son raisonnement (SHAP = <span className="mono" style={{color:'#F59E0B'}}>{features[0]?.value.toFixed(3)}</span>). <span style={{color:'#3B82F6',fontWeight:600}}>{features[1]?.label}</span> est le second facteur le plus influent ({features[1]?.value.toFixed(3)}). <span style={{color:'#3B82F6',fontWeight:600}}>{features[2]?.label}</span> contribue à hauteur de {features[2]?.value.toFixed(3)} en moyenne.
                  </div>
                </div>
              </div>
            </div>

            <div className="glass" style={{padding:14,borderRadius:14}}>
              <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:8}}>
                <span style={{fontSize:11,color:'#7F8FAD',letterSpacing:'.14em'}}>SHAP SUMMARY PLOT</span>
                <span className="mono" style={{fontSize:10,color:'#7F8FAD'}}>shap.summary_plot()</span>
              </div>
              <img
                src="http://localhost:8000/models/shap/summary-image"
                alt="SHAP summary plot"
                style={{width:'100%',borderRadius:8,border:'1px solid rgba(59,130,246,.15)'}}
                onError={e=>{e.target.style.display='none'; e.target.nextSibling.style.display='grid';}}
              />
              <div style={{display:'none',height:150,borderRadius:10,background:'repeating-linear-gradient(135deg,rgba(59,130,246,.04) 0 10px,rgba(59,130,246,.08) 10px 20px)',border:'1px dashed rgba(59,130,246,.25)',placeItems:'center'}}>
                <div style={{textAlign:'center'}}><div className="mono" style={{fontSize:11,color:'#7F8FAD'}}>Lancez l'API pour voir l'image</div></div>
              </div>
            </div>

            <div className="glass" style={{padding:14,borderRadius:14}}>
              <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:8}}>
                <span style={{fontSize:11,color:'#7F8FAD',letterSpacing:'.14em'}}>SHAP BAR PLOT</span>
                <span className="mono" style={{fontSize:10,color:'#7F8FAD'}}>shap.plots.bar()</span>
              </div>
              <img
                src="http://localhost:8000/models/shap/bar-image"
                alt="SHAP bar plot"
                style={{width:'100%',borderRadius:8,border:'1px solid rgba(59,130,246,.15)'}}
                onError={e=>{e.target.style.display='none'; e.target.nextSibling.style.display='grid';}}
              />
              <div style={{display:'none',height:150,borderRadius:10,background:'repeating-linear-gradient(135deg,rgba(59,130,246,.04) 0 10px,rgba(59,130,246,.08) 10px 20px)',border:'1px dashed rgba(59,130,246,.25)',placeItems:'center'}}>
                <div style={{textAlign:'center'}}><div className="mono" style={{fontSize:11,color:'#7F8FAD'}}>Lancez l'API pour voir l'image</div></div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Methodology */}
      <div className="glass" style={{borderRadius:16,overflow:'hidden'}}>
        <button onClick={()=>setExpanded(!expanded)} style={{width:'100%',padding:'16px 20px',background:'transparent',border:'none',color:'#E8F0FF',display:'flex',alignItems:'center',gap:14,cursor:'pointer',textAlign:'left'}}>
          <div style={{width:32,height:32,borderRadius:8,background:'rgba(139,92,246,.15)',border:'1px solid rgba(139,92,246,.4)',display:'grid',placeItems:'center',color:'#A78BFA'}}>
            <I.Info size={16}/>
          </div>
          <div style={{flex:1}}>
            <div className="grotesk" style={{fontSize:14,fontWeight:600}}>How SHAP works — methodology</div>
            <div style={{fontSize:11,color:'#7F8FAD',marginTop:2}}>{expanded?'click to collapse':'click to expand — game-theoretic foundation, computation cost, caveats'}</div>
          </div>
          <span style={{transform:expanded?'rotate(180deg)':'none',transition:'transform .2s'}}><I.ChevronDown size={18} stroke="#7F8FAD"/></span>
        </button>
        {expanded && (
          <div style={{padding:'4px 20px 20px',borderTop:'1px dashed rgba(59,130,246,.15)'}}>
            <table style={{width:'100%',borderCollapse:'collapse',fontSize:12.5,marginTop:10}}>
              <thead>
                <tr style={{textAlign:'left',color:'#7F8FAD',fontSize:10.5,letterSpacing:'.12em',borderBottom:'1px solid rgba(59,130,246,.15)'}}>
                  <th style={{padding:'10px 12px',width:'18%'}}>CONCEPT</th>
                  <th style={{padding:'10px 12px'}}>DEFINITION</th>
                  <th style={{padding:'10px 12px',width:'24%'}}>HERE</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ['Shapley value','Average marginal contribution of a feature across all possible coalitions, derived from cooperative game theory (Lloyd Shapley, 1953).','one phi per feature'],
                  ['TreeExplainer','Polynomial-time exact algorithm for tree ensembles (Lundberg et al., 2020). Replaces sampling-based KernelSHAP for XGBoost.',`${features.length} features analyzed`],
                  ['Mean |SHAP|','Aggregated absolute contribution — used for global importance ranking.',`${features[0]?.label} = ${features[0]?.value.toFixed(3)}`],
                  ['Additivity','Predictions are decomposable: y_hat = base_value + sum of phi_i — exact, not approximate.','base_value ≈ 0.148'],
                  ['Caveat','Correlated features split importance among themselves; interactions need shap.interaction_values.','handled via shap_feature_importance.csv'],
                ].map((row,i)=>(
                  <tr key={i} style={{borderBottom:i<4?'1px dashed rgba(59,130,246,.08)':'none'}}>
                    <td style={{padding:'10px 12px',color:'#A78BFA',fontWeight:600}}>{row[0]}</td>
                    <td style={{padding:'10px 12px',color:'#B7C5DD',lineHeight:1.5}}>{row[1]}</td>
                    <td style={{padding:'10px 12px',color:'#E8F0FF'}} className="mono">{row[2]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

window.PageShap = PageShap;
