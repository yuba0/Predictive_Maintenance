const API_BASE = 'http://localhost:8000';

function PageSimulator() {
  const { useState, useCallback } = React;
  const [machine, setMachine] = useState('cnc');
  const [mode, setMode] = useState('normal');
  const [vals, setVals] = useState({
    vibration_rms: 1.5,
    temperature_motor: 62,
    current_phase_avg: 8,
    pressure_level: 25,
    rpm: 1500,
    hours_since_maintenance: 120,
    ambient_temp: 13,
  });
  const [result, setResult] = useState({ prob: null, submitted: false });
  const [computing, setComputing] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [latency, setLatency] = useState(null);
  const [reqId] = useState(() => 'req_' + Math.random().toString(36).slice(2, 10));

  const MACHINE_MAP = { cnc: 'CNC', pump: 'Pump', compressor: 'Compressor', robot: 'Robotic Arm' };

  const onAnalyze = useCallback(async () => {
    setComputing(true);
    setApiError(null);
    const t0 = performance.now();
    try {
      const payload = {
        machine_type: MACHINE_MAP[machine],
        operating_mode: mode,
        vibration_rms: vals.vibration_rms,
        temperature_motor: vals.temperature_motor,
        current_phase_avg: vals.current_phase_avg,
        pressure_level: vals.pressure_level,
        rpm: vals.rpm,
        hours_since_maintenance: vals.hours_since_maintenance,
        ambient_temp: vals.ambient_temp,
      };
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setLatency(Math.round(performance.now() - t0));
      setResult({
        prob: data.failure_probability,
        predicted_class: data.predicted_class,
        risk_level: data.risk_level,
        threshold: data.threshold_used,
        model_name: data.model_name,
        recommendation: data.recommendation,
        submitted: true,
      });
    } catch (e) {
      const msg = e.message.toLowerCase().includes('fetch')
        ? 'API non disponible — lancez : python3 -m uvicorn api.main:app --port 8000'
        : e.message;
      setApiError(msg);
    } finally {
      setComputing(false);
    }
  }, [machine, mode, vals]);

  const sliders = [
    { key:'vibration_rms',           label:'Vibration RMS',          unit:'m/s²', min:0.35, max:10.0, step:0.05, normal:'0.5 – 2.0', danger:'> 5.0' },
    { key:'temperature_motor',       label:'Motor temperature',       unit:'°C',   min:28,   max:95,   step:1,    normal:'40 – 70',   danger:'> 85'  },
    { key:'current_phase_avg',       label:'Phase current avg.',      unit:'A',    min:0,    max:20,   step:0.5,  normal:'5 – 12',    danger:'> 15'  },
    { key:'pressure_level',          label:'Pressure level',          unit:'bar',  min:0,    max:50,   step:0.5,  normal:'10 – 30',   danger:'> 40'  },
    { key:'rpm',                     label:'Spindle RPM',             unit:'rpm',  min:0,    max:3000, step:25,   normal:'800 – 2000',danger:'> 2500'},
    { key:'hours_since_maintenance', label:'Hours since maintenance', unit:'h',    min:0,    max:600,  step:5,    normal:'< 200',     danger:'> 400' },
    { key:'ambient_temp',            label:'Ambient temperature',     unit:'°C',   min:8,    max:18,   step:0.5,  normal:'10 – 15',   danger:'> 16'  },
  ];

  const machines = [
    { id:'cnc',        name:'CNC Mill',       sub:'High-speed 5-axis',     G: MachineGlyph.cnc },
    { id:'pump',       name:'Hydraulic Pump', sub:'Variable displacement', G: MachineGlyph.pump },
    { id:'compressor', name:'Compressor',     sub:'Twin-screw',            G: MachineGlyph.compressor },
    { id:'robot',      name:'Robotic Arm',    sub:'6-axis articulated',    G: MachineGlyph.robot },
  ];

  const RISK_MAP = {
    'Faible':   { level:'FAIBLE',   color:'#06D6A0', glow:'rgba(6,214,160,.55)' },
    'Modéré':   { level:'MODÉRÉ',   color:'#FBBF24', glow:'rgba(251,191,36,.5)' },
    'Élevé':    { level:'ÉLEVÉ',    color:'#F59E0B', glow:'rgba(245,158,11,.6)' },
    'Critique': { level:'CRITIQUE', color:'#EF4444', glow:'rgba(239,68,68,.6)'  },
  };

  const probPct = result.prob ?? 0;
  const riskConfig = result.risk_level
    ? (RISK_MAP[result.risk_level] || RISK_MAP['Modéré'])
    : RISK_MAP['Faible'];
  const threshold = result.threshold ?? 0.5;
  const machineName = machines.find(m => m.id === machine)?.name ?? machine;

  return (
    <div style={{display:'grid',gridTemplateColumns:'1.05fr 1fr',gap:18,padding:'24px 28px 40px'}}>

      {/* LEFT — inputs */}
      <div className="glass" style={{padding:22,borderRadius:18,display:'flex',flexDirection:'column',gap:18}}>
        <div style={{display:'flex',alignItems:'flex-start',justifyContent:'space-between'}}>
          <div>
            <div style={{fontSize:10.5,color:'#7F8FAD',letterSpacing:'.16em'}}>STEP 01 · INPUT</div>
            <h2 className="grotesk" style={{margin:'4px 0 0',fontSize:20,fontWeight:700,letterSpacing:'-.02em'}}>Configure inference parameters</h2>
            <div style={{fontSize:12,color:'#B7C5DD',marginTop:4}}>Adjust the 7 sensor channels — the best trained model returns a real failure probability via API.</div>
          </div>
          <StatusBadge status="cyan" label="LIVE API"/>
        </div>

        {/* Machine selector */}
        <div>
          <div style={{fontSize:11,color:'#7F8FAD',letterSpacing:'.12em',marginBottom:8}}>MACHINE TYPE</div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:10}}>
            {machines.map(m=>{
              const active = machine===m.id;
              return (
                <button key={m.id} onClick={()=>setMachine(m.id)} style={{
                  position:'relative',padding:'14px 10px',borderRadius:12,
                  background: active ? 'linear-gradient(180deg, rgba(59,130,246,.18), rgba(0,180,216,.06))' : 'rgba(10,22,40,.5)',
                  border: active ? '1px solid rgba(59,130,246,.55)' : '1px solid rgba(59,130,246,.12)',
                  boxShadow: active ? '0 0 24px -6px rgba(59,130,246,.55)' : 'none',
                  color:'#E8F0FF', cursor:'pointer', display:'flex',flexDirection:'column',alignItems:'center',gap:6,
                  transition:'all .15s',
                }}>
                  <m.G size={36} color={active?'#3B82F6':'#7F8FAD'}/>
                  <div className="grotesk" style={{fontSize:12,fontWeight:600}}>{m.name}</div>
                  <div style={{fontSize:9.5,color:'#7F8FAD'}}>{m.sub}</div>
                  {active && <div style={{position:'absolute',top:8,right:8,width:6,height:6,borderRadius:99,background:'#3B82F6',boxShadow:'0 0 8px #3B82F6'}}/>}
                </button>
              );
            })}
          </div>
        </div>

        {/* Mode toggle */}
        <div>
          <div style={{fontSize:11,color:'#7F8FAD',letterSpacing:'.12em',marginBottom:8}}>OPERATING MODE</div>
          <div style={{display:'flex',gap:8}}>
            {[
              {id:'idle',   label:'Idle',   sub:'< 20% load',  col:'#06D6A0'},
              {id:'normal', label:'Normal', sub:'20–75% load', col:'#3B82F6'},
              {id:'peak',   label:'Peak',   sub:'> 75% load',  col:'#F59E0B'},
            ].map(o=>{
              const active = mode===o.id;
              return (
                <button key={o.id} onClick={()=>setMode(o.id)} style={{
                  flex:1,padding:'10px 12px',borderRadius:10,
                  background: active? `linear-gradient(180deg, ${o.col}28, ${o.col}08)` : 'rgba(10,22,40,.5)',
                  border: active? `1px solid ${o.col}88` : '1px solid rgba(59,130,246,.12)',
                  color:'#E8F0FF', cursor:'pointer', textAlign:'left',
                  boxShadow: active? `0 0 18px -4px ${o.col}88` : 'none',
                  transition:'all .15s',
                }}>
                  <div style={{display:'flex',alignItems:'center',gap:6}}>
                    <span style={{width:7,height:7,borderRadius:99,background:o.col,boxShadow:`0 0 8px ${o.col}`}}/>
                    <span className="grotesk" style={{fontWeight:600,fontSize:13}}>{o.label}</span>
                  </div>
                  <div style={{fontSize:10.5,color:'#7F8FAD',marginTop:2}}>{o.sub}</div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Sliders */}
        <div>
          <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:10}}>
            <div style={{fontSize:11,color:'#7F8FAD',letterSpacing:'.12em'}}>SENSOR CHANNELS · 7</div>
            <button onClick={()=>setVals({ vibration_rms:1.5, temperature_motor:62, current_phase_avg:8, pressure_level:25, rpm:1500, hours_since_maintenance:120, ambient_temp:13 })}
              style={{background:'transparent',border:'none',color:'#7F8FAD',fontSize:11,cursor:'pointer',display:'flex',alignItems:'center',gap:4}}>
              <I.Refresh size={11}/> Reset to nominal
            </button>
          </div>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12}}>
            {sliders.map(s=>{
              const v = vals[s.key];
              const t = (v-s.min)/(s.max-s.min);
              const isWarn = (s.danger.startsWith('>') && v > +s.danger.slice(2)) || (s.danger.startsWith('<') && v < +s.danger.slice(2));
              return (
                <div key={s.key} style={{background:'rgba(10,22,40,.5)',border:'1px solid rgba(59,130,246,.1)',borderRadius:10,padding:'10px 12px'}}>
                  <div style={{display:'flex',justifyContent:'space-between',alignItems:'baseline'}}>
                    <span className="mono" style={{fontSize:10.5,color:'#B7C5DD',letterSpacing:'.04em'}}>{s.key}</span>
                    <span className="grotesk tabular" style={{fontSize:15,fontWeight:700,color:isWarn?'#F59E0B':'#E8F0FF'}}>{typeof v==='number'&&v%1!==0?v.toFixed(2):v}<span style={{fontSize:10,color:'#566584',marginLeft:3}}>{s.unit}</span></span>
                  </div>
                  <div style={{position:'relative',height:6,background:'rgba(59,130,246,.08)',borderRadius:99,marginTop:8}}>
                    <div style={{position:'absolute',left:0,top:0,bottom:0,width:`${t*100}%`,background:`linear-gradient(90deg,#3B82F6,${isWarn?'#F59E0B':'#00B4D8'})`,borderRadius:99,boxShadow:`0 0 12px ${isWarn?'#F59E0B88':'#00B4D888'}`}}/>
                    <input type="range" min={s.min} max={s.max} step={s.step} value={v} onChange={e=>setVals({...vals,[s.key]:+e.target.value})}
                      style={{position:'absolute',inset:0,opacity:0,cursor:'pointer',width:'100%',height:'100%'}}/>
                    <div style={{position:'absolute',left:`calc(${t*100}% - 7px)`,top:-4,width:14,height:14,borderRadius:99,background:'#0A1628',border:`2px solid ${isWarn?'#F59E0B':'#3B82F6'}`,boxShadow:`0 0 8px ${isWarn?'#F59E0B':'#3B82F6'}`,pointerEvents:'none'}}/>
                  </div>
                  <div style={{display:'flex',justifyContent:'space-between',fontSize:9.5,color:'#566584',marginTop:6}}>
                    <span>nominal {s.normal}</span>
                    <span style={{color:isWarn?'#F59E0B':'#566584'}}>⚠ {s.danger}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {apiError && (
          <div style={{padding:'10px 14px',borderRadius:10,background:'rgba(239,68,68,.1)',border:'1px solid rgba(239,68,68,.35)',color:'#EF4444',fontSize:12,display:'flex',alignItems:'center',gap:10}}>
            <I.AlertTriangle size={14}/> {apiError}
          </div>
        )}

        <button onClick={onAnalyze} disabled={computing} style={{
          position:'relative',padding:'16px 22px',borderRadius:12,border:'none',cursor:computing?'not-allowed':'pointer',
          background:'linear-gradient(135deg, #3B82F6 0%, #00B4D8 100%)',
          color:'#fff',fontSize:14,fontWeight:700,letterSpacing:'.04em',
          boxShadow:'0 0 24px -2px rgba(59,130,246,.7), inset 0 1px 0 rgba(255,255,255,.2)',
          display:'flex',alignItems:'center',justifyContent:'center',gap:10,
          opacity:computing?0.8:1,
        }}>
          {computing ? <>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2"><circle cx="12" cy="12" r="9" strokeDasharray="20 40"><animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/></circle></svg>
            APPEL API EN COURS…
          </> : <>
            <I.Bolt size={16}/> ANALYZE — REAL MODEL INFERENCE
          </>}
        </button>
      </div>

      {/* RIGHT — result */}
      <div style={{display:'flex',flexDirection:'column',gap:14}}>

        {!result.submitted ? (
          <div className="glass" style={{padding:40,borderRadius:18,display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',gap:14,minHeight:280,textAlign:'center'}}>
            <div style={{width:60,height:60,borderRadius:16,background:'rgba(59,130,246,.12)',border:'1px solid rgba(59,130,246,.3)',display:'grid',placeItems:'center',color:'#3B82F6'}}>
              <I.Bolt size={28}/>
            </div>
            <div className="grotesk" style={{fontSize:18,fontWeight:600,color:'#E8F0FF'}}>Configure & Analyze</div>
            <div style={{fontSize:12,color:'#7F8FAD',maxWidth:280,lineHeight:1.6}}>Ajuste les capteurs à gauche, puis clique <strong style={{color:'#3B82F6'}}>ANALYZE</strong> pour obtenir une vraie prédiction via l'API FastAPI.</div>
          </div>
        ) : (
          <div className="glass" style={{padding:22,borderRadius:18,position:'relative',overflow:'hidden'}}>
            <div style={{position:'absolute',top:-20,right:-20,width:240,height:240,background:`radial-gradient(circle,${riskConfig.glow},transparent 60%)`,opacity:.4,pointerEvents:'none'}}/>
            <div style={{display:'flex',alignItems:'flex-start',justifyContent:'space-between'}}>
              <div>
                <div style={{fontSize:10.5,color:'#7F8FAD',letterSpacing:'.16em'}}>STEP 02 · RESULT</div>
                <h2 className="grotesk" style={{margin:'4px 0 0',fontSize:20,fontWeight:700,letterSpacing:'-.02em'}}>Failure probability · next 24h</h2>
                <div style={{fontSize:11,color:'#7F8FAD',marginTop:2}}>Model: <span className="mono" style={{color:'#3B82F6'}}>{result.model_name}</span></div>
              </div>
              <span style={{
                padding:'8px 14px',borderRadius:999,fontSize:11.5,fontWeight:700,letterSpacing:'.16em',
                background:`linear-gradient(180deg,${riskConfig.color}33,${riskConfig.color}0a)`,
                border:`1px solid ${riskConfig.color}88`,color:riskConfig.color,
                boxShadow:`0 0 24px -4px ${riskConfig.glow}`,
                display:'inline-flex',alignItems:'center',gap:8,
              }}>
                <span style={{width:7,height:7,borderRadius:99,background:riskConfig.color,boxShadow:`0 0 8px ${riskConfig.color}`}} className="pulse"/>
                RISK · {riskConfig.level}
              </span>
            </div>
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',alignItems:'center',gap:14,marginTop:10}}>
              <ArcGauge value={probPct} label="P(failure ≤ 24h)" sub={result.model_name||'Model'} size={300}/>
              <div style={{display:'flex',flexDirection:'column',gap:10}}>
                <div style={{padding:'12px 14px',background:'rgba(10,22,40,.5)',borderRadius:12,border:'1px solid rgba(59,130,246,.18)'}}>
                  <div style={{fontSize:10,color:'#7F8FAD',letterSpacing:'.12em'}}>DECISION THRESHOLD</div>
                  <div style={{display:'flex',alignItems:'baseline',gap:6,marginTop:3}}>
                    <span className="grotesk tabular" style={{fontSize:22,fontWeight:700}}>{threshold.toFixed(2)}</span>
                    <span style={{fontSize:11,color:'#7F8FAD'}}>· optimised on val set</span>
                  </div>
                </div>
                <div style={{padding:'12px 14px',background:'rgba(10,22,40,.5)',borderRadius:12,border:'1px solid rgba(59,130,246,.18)'}}>
                  <div style={{fontSize:10,color:'#7F8FAD',letterSpacing:'.12em'}}>PREDICTED CLASS</div>
                  <div style={{display:'flex',alignItems:'baseline',gap:6,marginTop:3}}>
                    <span className="grotesk" style={{fontSize:18,fontWeight:700,color:result.predicted_class===1?'#EF4444':'#06D6A0'}}>
                      {result.predicted_class===1?'FAILURE':'NO FAILURE'}
                    </span>
                  </div>
                  <div style={{fontSize:11,color:'#B7C5DD',marginTop:3}}>confidence {(Math.abs(probPct-0.5)*200).toFixed(1)}%</div>
                </div>
                <div style={{padding:'12px 14px',background:'rgba(10,22,40,.5)',borderRadius:12,border:'1px solid rgba(59,130,246,.18)'}}>
                  <div style={{fontSize:10,color:'#7F8FAD',letterSpacing:'.12em'}}>EXPECTED RUL (estimate)</div>
                  <div style={{display:'flex',alignItems:'baseline',gap:6,marginTop:3}}>
                    <span className="grotesk tabular" style={{fontSize:22,fontWeight:700,color:'#00B4D8'}}>{Math.round((1-probPct)*820+8)}<span style={{fontSize:12,color:'#7F8FAD',marginLeft:3}}>h</span></span>
                    <span style={{fontSize:11,color:'#7F8FAD'}}>· ≈ {((1-probPct)*820/24).toFixed(1)} days</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {result.submitted && (
          <div className="glass" style={{padding:18,borderRadius:14,borderLeft:`3px solid ${riskConfig.color}`,boxShadow:`inset 4px 0 12px -8px ${riskConfig.glow}`}}>
            <div style={{display:'flex',alignItems:'flex-start',gap:14}}>
              <div style={{width:42,height:42,borderRadius:10,background:`${riskConfig.color}22`,border:`1px solid ${riskConfig.color}66`,display:'grid',placeItems:'center',color:riskConfig.color,flexShrink:0}}>
                <I.Wrench size={20}/>
              </div>
              <div style={{flex:1}}>
                <div style={{fontSize:11,color:'#7F8FAD',letterSpacing:'.12em'}}>RECOMMENDED ACTION · API RESPONSE</div>
                <div className="grotesk" style={{fontSize:14,fontWeight:600,marginTop:2,color:'#E8F0FF'}}>{result.recommendation}</div>
                <div style={{display:'flex',gap:8,marginTop:10}}>
                  <button style={{padding:'6px 12px',borderRadius:8,border:`1px solid ${riskConfig.color}`,background:`${riskConfig.color}22`,color:riskConfig.color,fontSize:11,fontWeight:600,cursor:'pointer'}}>Dispatch ticket</button>
                  <button style={{padding:'6px 12px',borderRadius:8,border:'1px solid rgba(59,130,246,.2)',background:'rgba(10,22,40,.5)',color:'#B7C5DD',fontSize:11,cursor:'pointer'}}>Add to watchlist</button>
                  <button style={{padding:'6px 12px',borderRadius:8,border:'1px solid rgba(59,130,246,.2)',background:'rgba(10,22,40,.5)',color:'#B7C5DD',fontSize:11,cursor:'pointer'}}>Open log</button>
                </div>
              </div>
            </div>
          </div>
        )}

        {result.submitted && (
          <div className="glass" style={{padding:18,borderRadius:14}}>
            <SectionTitle accent="#00B4D8">Inference summary · real API response</SectionTitle>
            <table style={{width:'100%',borderCollapse:'collapse',fontSize:12}}>
              <tbody>
                {[
                  ['Request ID',        <span className="mono" style={{color:'#00B4D8'}}>{reqId}</span>],
                  ['Machine type',       machineName],
                  ['Operating mode',     mode.toUpperCase()],
                  ['Model',             <span className="mono" style={{color:'#3B82F6'}}>{result.model_name}</span>],
                  ['Decision threshold',<span className="mono">{threshold.toFixed(4)}</span>],
                  ['Probability',       <span className="mono" style={{color:riskConfig.color,fontWeight:700}}>{(probPct*100).toFixed(2)} %</span>],
                  ['Predicted class',   <span style={{color:result.predicted_class===1?'#EF4444':'#06D6A0',fontWeight:700}}>{result.predicted_class===1?'1 · failure':'0 · no failure'}</span>],
                  ['Inference latency', <span className="mono">{latency??'—'} ms</span>],
                ].map(([k,v],i)=>(
                  <tr key={i} style={{borderBottom:i<7?'1px dashed rgba(59,130,246,.1)':'none'}}>
                    <td style={{padding:'8px 0',color:'#7F8FAD'}}>{k}</td>
                    <td style={{padding:'8px 0',color:'#E8F0FF',textAlign:'right'}}>{v}</td>
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

window.PageSimulator = PageSimulator;
