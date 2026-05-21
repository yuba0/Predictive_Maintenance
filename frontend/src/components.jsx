// Shared chrome: Sidebar, Topbar, KPI Card, MachineCard, Badge, etc.

window.useStateC = React.useState;
window.useEffectC = React.useEffect;
const { useState: useStateC, useEffect: useEffectC } = React;

/* ------------------------------- Sidebar ------------------------------- */
function Sidebar({ active, onNav }) {
  const items = [
    { id:'overview',    label:'Live Overview',     icon:<I.Overview/> },
    { id:'simulator',   label:'Prediction Simulator',icon:<I.Simulator/> },
    { id:'models',      label:'Model Performance',  icon:<I.Models/> },
    { id:'shap',        label:'Interpretability',   icon:<I.Shap/> },
  ];
  return (
    <aside style={{
      width:260, minWidth:260, height:'100vh', position:'sticky', top:0,
      background:'linear-gradient(180deg, #0A1628 0%, #060E1C 100%)',
      borderRight:'1px solid rgba(59,130,246,.12)',
      display:'flex', flexDirection:'column', padding:'18px 14px 14px',
      zIndex:5,
    }}>
      {/* Logo */}
      <div style={{display:'flex',alignItems:'center',gap:10,padding:'4px 8px 16px'}}>
        <div className="float"><I.Logo size={32}/></div>
        <div>
          <div className="grotesk" style={{fontWeight:700,fontSize:16,letterSpacing:'-0.01em'}}>PredictMaint</div>
          <div className="mono" style={{fontSize:9.5,color:'#7F8FAD',letterSpacing:'.14em',textTransform:'uppercase'}}>AI · Asset Health</div>
        </div>
      </div>

      <div className="hairline" style={{margin:'2px 6px 12px',opacity:.5}}/>

      {/* Nav */}
      <nav style={{display:'flex',flexDirection:'column',gap:4,padding:'0 2px'}}>
        <div style={{fontSize:9.5,letterSpacing:'.16em',color:'#566584',padding:'8px 10px 6px'}}>WORKSPACE</div>
        {items.map(it=>{
          const isActive = active===it.id;
          return (
            <button key={it.id} onClick={()=>onNav(it.id)} style={{
              position:'relative', display:'flex', alignItems:'center', gap:11, padding:'10px 12px',
              background: isActive ? 'linear-gradient(90deg, rgba(59,130,246,.16), rgba(59,130,246,.02))' : 'transparent',
              border:'none', color: isActive ? '#E8F0FF' : '#B7C5DD',
              borderRadius:10, fontSize:13, fontWeight:isActive?600:500, cursor:'pointer',
              textAlign:'left', transition:'all .2s',
              boxShadow: isActive ? '0 0 24px -8px rgba(59,130,246,.5)' : 'none',
            }}>
              {isActive && <span style={{position:'absolute',left:-2,top:8,bottom:8,width:3,borderRadius:2,background:'#3B82F6',boxShadow:'0 0 12px #3B82F6'}}/>}
              <span style={{color:isActive?'#3B82F6':'#7F8FAD',display:'flex'}}>{it.icon}</span>
              <span style={{flex:1}}>{it.label}</span>
              {isActive && <I.ChevronRight size={14}/>}
            </button>
          );
        })}
      </nav>

      <div style={{flex:1}}/>

      {/* Model status card */}
      <div className="glass" style={{padding:12,marginBottom:10,borderRadius:12}}>
        <div style={{display:'flex',alignItems:'center',gap:8,marginBottom:8}}>
          <span style={{width:8,height:8,background:'#06D6A0',color:'#06D6A0'}} className="pulse"/>
          <span style={{fontSize:10,color:'#7F8FAD',letterSpacing:'.14em'}}>MODEL ONLINE</span>
        </div>
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'baseline'}}>
          <span className="grotesk" style={{fontWeight:600,fontSize:13}}>XGBoost v2.4.1</span>
          <span className="mono" style={{fontSize:10,color:'#7F8FAD'}}>★ deployed</span>
        </div>
        <div style={{display:'flex',justifyContent:'space-between',marginTop:8,fontSize:10.5,color:'#B7C5DD'}}>
          <span>Threshold</span>
          <span className="mono" style={{color:'#00B4D8'}}>0.42</span>
        </div>
        <div style={{display:'flex',justifyContent:'space-between',marginTop:3,fontSize:10.5,color:'#B7C5DD'}}>
          <span>Latency p95</span>
          <span className="mono" style={{color:'#E8F0FF'}}>38 ms</span>
        </div>
      </div>

      {/* Dataset stats */}
      <div className="glass" style={{padding:12,marginBottom:10,borderRadius:12}}>
        <div style={{fontSize:10,color:'#7F8FAD',letterSpacing:'.14em',marginBottom:8}}>DATASET</div>
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:10}}>
          <div>
            <div className="mono grotesk tabular" style={{fontSize:16,fontWeight:700}}>24,042</div>
            <div style={{fontSize:9.5,color:'#7F8FAD'}}>observations</div>
          </div>
          <div>
            <div className="mono grotesk tabular" style={{fontSize:16,fontWeight:700,color:'#F59E0B'}}>14.8%</div>
            <div style={{fontSize:9.5,color:'#7F8FAD'}}>failure rate</div>
          </div>
          <div>
            <div className="mono grotesk tabular" style={{fontSize:16,fontWeight:700}}>4</div>
            <div style={{fontSize:9.5,color:'#7F8FAD'}}>machine types</div>
          </div>
          <div>
            <div className="mono grotesk tabular" style={{fontSize:16,fontWeight:700,color:'#06D6A0'}}>7</div>
            <div style={{fontSize:9.5,color:'#7F8FAD'}}>sensor channels</div>
          </div>
        </div>
      </div>

      {/* Branding */}
      <div style={{padding:'10px 8px 4px',display:'flex',alignItems:'center',justifyContent:'space-between',fontSize:10.5,color:'#566584'}}>
        <span className="mono" style={{letterSpacing:'.16em'}}>EFREI · M2 DS</span>
        <span className="mono">v 2.4.1</span>
      </div>
    </aside>
  );
}

/* ------------------------------- Topbar ------------------------------- */
function Topbar({ title, sub, breadcrumb }) {
  const now = new Date('2026-05-21T09:42:00');
  const ts = now.toLocaleString('en-GB',{weekday:'short',day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'});
  return (
    <header style={{
      display:'flex',alignItems:'center',gap:18,padding:'18px 28px',
      borderBottom:'1px solid rgba(59,130,246,.08)',
      background:'linear-gradient(180deg, rgba(10,22,40,.7), rgba(10,22,40,.2))',
      backdropFilter:'blur(10px)',
      position:'sticky',top:0,zIndex:4,
    }}>
      <div style={{flex:1,minWidth:0}}>
        <div style={{fontSize:10.5,color:'#566584',letterSpacing:'.16em',marginBottom:4}}>{breadcrumb}</div>
        <div style={{display:'flex',alignItems:'baseline',gap:14}}>
          <h1 className="grotesk" style={{margin:0,fontSize:24,fontWeight:700,letterSpacing:'-0.02em'}}>{title}</h1>
          {sub && <span style={{fontSize:13,color:'#7F8FAD'}}>{sub}</span>}
        </div>
      </div>

      {/* Search */}
      <div style={{display:'flex',alignItems:'center',gap:10,padding:'8px 12px',background:'rgba(21,40,71,.5)',border:'1px solid rgba(59,130,246,.12)',borderRadius:10,width:260}}>
        <I.Search size={14} stroke="#7F8FAD"/>
        <input placeholder="Search machines, sensors, alerts…" style={{background:'transparent',border:'none',outline:'none',color:'#E8F0FF',fontSize:12.5,flex:1,fontFamily:'inherit'}}/>
        <span className="mono" style={{fontSize:9.5,color:'#566584',padding:'2px 6px',border:'1px solid rgba(59,130,246,.18)',borderRadius:4}}>⌘K</span>
      </div>

      {/* model status pill */}
      <div className="glass" style={{display:'flex',alignItems:'center',gap:10,padding:'7px 12px',borderRadius:999}}>
        <span style={{width:7,height:7,background:'#06D6A0',color:'#06D6A0'}} className="pulse"/>
        <span style={{fontSize:11.5,color:'#B7C5DD'}}>Inference</span>
        <span className="mono" style={{fontSize:11.5,color:'#E8F0FF',fontWeight:600}}>XGBoost · 0.42</span>
      </div>

      {/* date */}
      <div className="mono" style={{display:'flex',alignItems:'center',gap:8,padding:'8px 12px',background:'rgba(21,40,71,.4)',border:'1px solid rgba(59,130,246,.1)',borderRadius:10,fontSize:11.5,color:'#B7C5DD'}}>
        <I.Calendar size={13} stroke="#7F8FAD"/> {ts}
      </div>

      {/* notifications */}
      <button style={{position:'relative',width:38,height:38,border:'1px solid rgba(59,130,246,.12)',background:'rgba(21,40,71,.4)',borderRadius:10,color:'#E8F0FF',cursor:'pointer',display:'grid',placeItems:'center'}}>
        <I.Bell size={16}/>
        <span style={{position:'absolute',top:6,right:6,width:8,height:8,background:'#EF4444',color:'#EF4444',borderRadius:99}} className="pulse"/>
      </button>

      {/* avatar */}
      <div style={{display:'flex',alignItems:'center',gap:10,padding:'4px 10px 4px 4px',background:'rgba(21,40,71,.4)',border:'1px solid rgba(59,130,246,.12)',borderRadius:999}}>
        <div style={{width:30,height:30,borderRadius:999,background:'linear-gradient(135deg,#3B82F6,#00B4D8)',display:'grid',placeItems:'center',color:'#fff',fontSize:11.5,fontWeight:700}}>LD</div>
        <div style={{display:'flex',flexDirection:'column'}}>
          <span style={{fontSize:11.5,fontWeight:600}}>L. Dubois</span>
          <span style={{fontSize:9.5,color:'#7F8FAD'}}>Maint. Manager</span>
        </div>
      </div>
    </header>
  );
}

/* ------------------------------- Badge ------------------------------- */
function StatusBadge({ status='ok', label, size='md', pulse=false }) {
  const map = {
    ok: { fg:'#06D6A0', bg:'rgba(6,214,160,.12)', bd:'rgba(6,214,160,.3)', text:'OPERATIONAL' },
    warn: { fg:'#F59E0B', bg:'rgba(245,158,11,.12)', bd:'rgba(245,158,11,.3)', text:'WARNING' },
    alert: { fg:'#EF4444', bg:'rgba(239,68,68,.12)', bd:'rgba(239,68,68,.3)', text:'ALERT' },
    crit: { fg:'#7B0000', bg:'rgba(123,0,0,.25)', bd:'rgba(239,68,68,.55)', text:'CRITICAL' },
    info: { fg:'#3B82F6', bg:'rgba(59,130,246,.12)', bd:'rgba(59,130,246,.3)', text:'INFO' },
    cyan: { fg:'#00B4D8', bg:'rgba(0,180,216,.12)', bd:'rgba(0,180,216,.3)', text:'ACTIVE' },
  };
  const s = map[status]||map.info;
  const sz = size==='sm'?{padY:3,padX:8,font:9.5,dot:6}:{padY:5,padX:10,font:10.5,dot:7};
  return (
    <span style={{
      display:'inline-flex',alignItems:'center',gap:6,
      padding:`${sz.padY}px ${sz.padX}px`,
      background:s.bg,border:`1px solid ${s.bd}`,color:s.fg,
      borderRadius:999,fontSize:sz.font,fontWeight:600,letterSpacing:'.1em',
      boxShadow:status==='crit'||status==='alert'?`0 0 18px -4px ${s.fg}66`:'none',
    }}>
      <span style={{width:sz.dot,height:sz.dot,borderRadius:99,background:s.fg,color:s.fg}} className={pulse?'pulse':''}/>
      {label||s.text}
    </span>
  );
}

/* ------------------------------- KPI Card ------------------------------- */
function KPICard({ label, value, unit, delta, deltaLabel, accent='blue', icon, sub, valueColor }) {
  const accentMap = {
    blue: { glow:'rgba(59,130,246,.5)', strip:'#3B82F6' },
    cyan: { glow:'rgba(0,180,216,.5)', strip:'#00B4D8' },
    green:{ glow:'rgba(6,214,160,.5)', strip:'#06D6A0' },
    amber:{ glow:'rgba(245,158,11,.5)', strip:'#F59E0B' },
    red:  { glow:'rgba(239,68,68,.55)', strip:'#EF4444' },
  };
  const a = accentMap[accent];
  return (
    <div className="glass fade-up" style={{
      position:'relative',padding:18,borderRadius:16,overflow:'hidden',
      boxShadow:`0 0 0 1px ${a.glow.replace('.5','.18')}, 0 0 40px -16px ${a.glow}, 0 12px 32px -16px rgba(0,0,0,.5)`,
    }}>
      {/* corner glow */}
      <div style={{position:'absolute',top:-30,right:-30,width:140,height:140,background:`radial-gradient(circle, ${a.glow}, transparent 60%)`,opacity:.6,filter:'blur(10px)',pointerEvents:'none'}}/>
      <div style={{position:'absolute',left:0,top:18,bottom:18,width:2,background:a.strip,borderRadius:2,boxShadow:`0 0 12px ${a.glow}`}}/>

      <div style={{display:'flex',alignItems:'flex-start',justifyContent:'space-between'}}>
        <div style={{display:'flex',alignItems:'center',gap:10}}>
          {icon && <span style={{color:a.strip,opacity:.9,filter:`drop-shadow(0 0 6px ${a.glow})`}}>{icon}</span>}
          <span style={{fontSize:10.5,color:'#7F8FAD',letterSpacing:'.14em',textTransform:'uppercase'}}>{label}</span>
        </div>
        {delta!==undefined && (
          <span style={{
            display:'inline-flex',alignItems:'center',gap:3,fontSize:10.5,
            color: delta>=0?'#06D6A0':'#EF4444', fontWeight:600,
            padding:'3px 6px', borderRadius:6,
            background: delta>=0?'rgba(6,214,160,.1)':'rgba(239,68,68,.1)',
          }}>
            {delta>=0?<I.TrendUp size={11}/>:<I.TrendDown size={11}/>}
            {delta>=0?'+':''}{delta}% {deltaLabel}
          </span>
        )}
      </div>

      <div style={{display:'flex',alignItems:'baseline',gap:6,marginTop:14}}>
        <span className="grotesk tabular" style={{fontSize:42,fontWeight:700,letterSpacing:'-0.02em',color: valueColor||'#E8F0FF',lineHeight:1,filter:`drop-shadow(0 0 12px ${a.glow})`}}>{value}</span>
        {unit && <span style={{fontSize:14,color:'#7F8FAD',fontWeight:500}}>{unit}</span>}
      </div>
      {sub && <div style={{marginTop:8,fontSize:11.5,color:'#B7C5DD'}}>{sub}</div>}
    </div>
  );
}

/* ------------------------------- Machine Card ------------------------------- */
function MachineCard({ name, code, type, status, last, spark, sensors, units }) {
  const map = {
    ok: { color:'#06D6A0', label:'OPERATIONAL' },
    warn:{ color:'#F59E0B', label:'DEGRADED' },
    alert:{ color:'#EF4444', label:'AT RISK' },
    crit:{ color:'#7B0000', label:'CRITICAL' },
  };
  const s = map[status];
  const glyphs = { cnc: MachineGlyph.cnc, pump: MachineGlyph.pump, compressor: MachineGlyph.compressor, robot: MachineGlyph.robot };
  const G = glyphs[type];
  return (
    <div className="glass" style={{padding:16,borderRadius:14,position:'relative',overflow:'hidden'}}>
      <div style={{position:'absolute',top:0,left:0,right:0,height:2,background:`linear-gradient(90deg, transparent, ${s.color}, transparent)`,opacity:.7}}/>

      <div style={{display:'flex',alignItems:'flex-start',gap:12}}>
        <div style={{width:60,height:60,borderRadius:12,background:`linear-gradient(135deg, rgba(${status==='ok'?'6,214,160':status==='warn'?'245,158,11':'239,68,68'},.18), rgba(59,130,246,.05))`,border:'1px solid rgba(59,130,246,.14)',display:'grid',placeItems:'center',color:s.color}}>
          <G color={s.color}/>
        </div>
        <div style={{flex:1,minWidth:0}}>
          <div style={{display:'flex',alignItems:'center',gap:8}}>
            <span style={{width:8,height:8,background:s.color,color:s.color,borderRadius:99}} className="pulse"/>
            <span style={{fontSize:10.5,color:s.color,letterSpacing:'.14em',fontWeight:700}}>{s.label}</span>
          </div>
          <div className="grotesk" style={{fontSize:15,fontWeight:700,marginTop:4,letterSpacing:'-.01em'}}>{name}</div>
          <div className="mono" style={{fontSize:10.5,color:'#7F8FAD',marginTop:1}}>{code}</div>
        </div>
        <div style={{textAlign:'right'}}>
          <Sparkline data={spark} color={s.color} width={92} height={32}/>
          <div className="mono" style={{fontSize:9.5,color:'#7F8FAD',marginTop:2}}>last 24h</div>
        </div>
      </div>

      <div style={{marginTop:12,display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:8}}>
        {sensors.map((v,i)=>(
          <div key={i} style={{background:'rgba(10,22,40,.45)',border:'1px solid rgba(59,130,246,.1)',borderRadius:8,padding:'7px 9px'}}>
            <div style={{fontSize:9,color:'#7F8FAD',letterSpacing:'.08em'}}>{v.label}</div>
            <div className="mono tabular" style={{fontSize:13,fontWeight:600,color:v.warn?'#F59E0B':'#E8F0FF',marginTop:1}}>{v.value}<span style={{fontSize:9.5,color:'#566584',marginLeft:2}}>{v.unit}</span></div>
          </div>
        ))}
      </div>

      <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginTop:12,paddingTop:12,borderTop:'1px dashed rgba(59,130,246,.1)'}}>
        <span style={{fontSize:10.5,color:'#7F8FAD'}}>RUL est.</span>
        <div style={{display:'flex',alignItems:'center',gap:8}}>
          <span className="mono grotesk tabular" style={{fontSize:14,fontWeight:700,color:status==='ok'?'#06D6A0':status==='warn'?'#F59E0B':'#EF4444'}}>{last.rul}h</span>
          <span style={{fontSize:10.5,color:'#7F8FAD'}}>P(fail) <span className="mono" style={{color:'#E8F0FF',fontWeight:600,marginLeft:4}}>{last.prob}%</span></span>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------- Section Title ------------------------------- */
function SectionTitle({ children, accent, action }) {
  return (
    <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:14}}>
      <div style={{display:'flex',alignItems:'center',gap:10}}>
        <div style={{width:3,height:18,background:accent||'#3B82F6',borderRadius:2,boxShadow:`0 0 10px ${accent||'#3B82F6'}`}}/>
        <h2 className="grotesk" style={{margin:0,fontSize:14.5,fontWeight:600,letterSpacing:'-0.01em',color:'#E8F0FF'}}>{children}</h2>
      </div>
      {action}
    </div>
  );
}

function Pill({ children, active, onClick, color='#3B82F6' }) {
  return (
    <button onClick={onClick} style={{
      padding:'6px 12px',borderRadius:999,fontSize:11.5,fontWeight:600,letterSpacing:'.04em',
      background: active ? `linear-gradient(180deg, ${color}33, ${color}11)` : 'rgba(10,22,40,.4)',
      border: active ? `1px solid ${color}88` : '1px solid rgba(59,130,246,.12)',
      color: active ? '#E8F0FF' : '#B7C5DD',
      boxShadow: active ? `0 0 18px -4px ${color}77` : 'none',
      cursor:'pointer', transition:'all .15s',
    }}>{children}</button>
  );
}

Object.assign(window, { Sidebar, Topbar, StatusBadge, KPICard, MachineCard, SectionTitle, Pill });
