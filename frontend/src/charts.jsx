// Hand-crafted SVG charts tuned to the PredictMaint design language.

const { useMemo, useState, useEffect, useRef } = React;

/* ----------------------- helpers ----------------------- */
const range = (n)=>Array.from({length:n},(_,i)=>i);
const lerp = (a,b,t)=>a+(b-a)*t;
const clamp = (x,a,b)=>Math.max(a,Math.min(b,x));
const fmtNum = (v, d=0)=> Number(v).toLocaleString('en-US', {minimumFractionDigits:d, maximumFractionDigits:d});

/* ----------------------- Sparkline ----------------------- */
function Sparkline({ data, color="#3B82F6", width=120, height=36, fillOpacity=.18 }) {
  const min = Math.min(...data), max = Math.max(...data);
  const pad = 2;
  const w = width, h = height;
  const stepX = (w - pad*2) / (data.length - 1);
  const pts = data.map((v,i)=>[pad+i*stepX, h-pad - ((v-min)/(max-min||1))*(h-pad*2)]);
  const d = pts.map((p,i)=>(i===0?'M':'L')+p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' ');
  const area = d + ` L ${w-pad} ${h-pad} L ${pad} ${h-pad} Z`;
  const id = useMemo(()=>'sp-'+Math.random().toString(36).slice(2,7),[]);
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity={fillOpacity}/>
          <stop offset="1" stopColor={color} stopOpacity="0"/>
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${id})`} stroke="none"/>
      <path d={d} fill="none" stroke={color} strokeWidth="1.4"/>
      <circle cx={pts[pts.length-1][0]} cy={pts[pts.length-1][1]} r="2.2" fill={color}/>
    </svg>
  );
}

/* ----------------------- Bar chart (vertical / horizontal) ----------------------- */
function BarChart({ data, labels, color="#3B82F6", height=180, width=440, maxValue, format=(v)=>v, valueLabel=true }) {
  const max = maxValue ?? Math.max(...data) * 1.15;
  const padL=36,padR=12,padT=10,padB=28;
  const W = width - padL - padR, H = height - padT - padB;
  const bw = W / data.length * 0.55;
  const step = W / data.length;
  const id = useMemo(()=>'bc-'+Math.random().toString(36).slice(2,7),[]);
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity=".95"/>
          <stop offset="1" stopColor={color} stopOpacity=".25"/>
        </linearGradient>
      </defs>
      {/* gridlines */}
      {[0,.25,.5,.75,1].map((t,i)=>{
        const y = padT + H*(1-t);
        return <g key={i}>
          <line x1={padL} x2={width-padR} y1={y} y2={y} stroke="rgba(59,130,246,.08)"/>
          <text x={padL-6} y={y+3} fontSize="9" textAnchor="end" fill="#7F8FAD" className="mono">{format(max*t)}</text>
        </g>;
      })}
      {data.map((v,i)=>{
        const h = (v/max)*H;
        const x = padL + step*i + (step-bw)/2;
        const y = padT + (H - h);
        return (
          <g key={i}>
            <rect x={x} y={y} width={bw} height={h} rx="3" fill={`url(#${id})`}/>
            <rect x={x} y={y} width={bw} height="2" rx="1" fill={color}/>
            {valueLabel && <text x={x+bw/2} y={y-4} fontSize="10" textAnchor="middle" fill="#E8F0FF" className="mono" fontWeight="600">{format(v)}</text>}
            <text x={x+bw/2} y={height-padB+14} fontSize="10" textAnchor="middle" fill="#7F8FAD">{labels[i]}</text>
          </g>
        );
      })}
    </svg>
  );
}

/* Grouped bar chart */
function GroupedBars({ groups, series, height=200, width=560, format=(v)=>v.toFixed(2) }){
  // groups: ['Recall','F1',...], series: [{name, color, values:[]}]
  const all = series.flatMap(s=>s.values);
  const max = Math.max(...all) * 1.12;
  const padL=36,padR=12,padT=18,padB=30;
  const W=width-padL-padR, H=height-padT-padB;
  const groupStep = W/groups.length;
  const bw = (groupStep*0.7)/series.length;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      {[0,.25,.5,.75,1].map((t,i)=>{
        const y=padT+H*(1-t);
        return <g key={i}>
          <line x1={padL} x2={width-padR} y1={y} y2={y} stroke="rgba(59,130,246,.07)"/>
          <text x={padL-6} y={y+3} fontSize="9" textAnchor="end" fill="#7F8FAD" className="mono">{(max*t).toFixed(2)}</text>
        </g>;
      })}
      {groups.map((g,gi)=>(
        <g key={gi}>
          <text x={padL + groupStep*gi + groupStep/2} y={height-padB+16} fontSize="10.5" textAnchor="middle" fill="#B7C5DD" fontWeight="500">{g}</text>
          {series.map((s,si)=>{
            const v=s.values[gi];
            const h=(v/max)*H;
            const x=padL+groupStep*gi + (groupStep*0.15) + bw*si;
            const y=padT+(H-h);
            return <g key={si}>
              <rect x={x} y={y} width={bw-2} height={h} rx="2" fill={s.color} opacity=".95"/>
            </g>;
          })}
        </g>
      ))}
    </svg>
  );
}

/* ----------------------- Donut ----------------------- */
function Donut({ slices, size=180, thickness=22, centerLabel, centerSub }) {
  const r = (size-thickness)/2;
  const cx = size/2, cy = size/2;
  const total = slices.reduce((a,s)=>a+s.value,0);
  let acc = -Math.PI/2;
  const seg = slices.map(s=>{
    const ang = (s.value/total)*Math.PI*2;
    const start = acc, end = acc+ang;
    acc = end;
    const x1 = cx + r*Math.cos(start), y1 = cy + r*Math.sin(start);
    const x2 = cx + r*Math.cos(end), y2 = cy + r*Math.sin(end);
    const large = ang>Math.PI?1:0;
    return { d: `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`, color: s.color, label: s.label, value: s.value, pct: (s.value/total*100) };
  });
  return (
    <div style={{display:'flex',alignItems:'center',gap:18}}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(59,130,246,.08)" strokeWidth={thickness}/>
        {seg.map((s,i)=>(
          <path key={i} d={s.d} fill="none" stroke={s.color} strokeWidth={thickness} strokeLinecap="butt"/>
        ))}
        <text x={cx} y={cy-2} textAnchor="middle" fontSize="22" fontWeight="700" fill="#E8F0FF" className="grotesk tabular">{centerLabel}</text>
        <text x={cx} y={cy+16} textAnchor="middle" fontSize="10" fill="#7F8FAD" letterSpacing=".08em">{centerSub}</text>
      </svg>
      <div style={{display:'flex',flexDirection:'column',gap:8,minWidth:160}}>
        {slices.map((s,i)=>(
          <div key={i} style={{display:'flex',alignItems:'center',gap:8,fontSize:12}}>
            <span style={{width:9,height:9,borderRadius:2,background:s.color,display:'inline-block',boxShadow:`0 0 8px ${s.color}66`}}/>
            <span style={{color:'#B7C5DD',flex:1}}>{s.label}</span>
            <span className="mono" style={{color:'#E8F0FF',fontWeight:600}}>{(s.value/total*100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ----------------------- Area / Line over time ----------------------- */
function AreaChart({ data, labels, color="#3B82F6", color2="#00B4D8", height=200, width=860, threshold }) {
  // data: number[]
  const max = Math.max(...data, threshold||0)*1.15;
  const padL=42,padR=14,padT=14,padB=26;
  const W=width-padL-padR, H=height-padT-padB;
  const stepX = W/(data.length-1);
  const pts = data.map((v,i)=>[padL+i*stepX, padT+H-(v/max)*H]);
  const d = pts.map((p,i)=>(i===0?'M':'L')+p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' ');
  const area = d + ` L ${padL+W} ${padT+H} L ${padL} ${padT+H} Z`;
  const id1 = useMemo(()=>'ar-'+Math.random().toString(36).slice(2,7),[]);
  const id2 = useMemo(()=>'lg-'+Math.random().toString(36).slice(2,7),[]);
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <defs>
        <linearGradient id={id1} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity=".35"/>
          <stop offset="1" stopColor={color} stopOpacity="0"/>
        </linearGradient>
        <linearGradient id={id2} x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor={color}/>
          <stop offset="1" stopColor={color2}/>
        </linearGradient>
      </defs>
      {[0,.25,.5,.75,1].map((t,i)=>{
        const y=padT+H*(1-t);
        return <g key={i}>
          <line x1={padL} x2={width-padR} y1={y} y2={y} stroke="rgba(59,130,246,.07)"/>
          <text x={padL-6} y={y+3} fontSize="9" textAnchor="end" fill="#7F8FAD" className="mono">{Math.round(max*t)}</text>
        </g>;
      })}
      {threshold !== undefined && (()=>{
        const y = padT+H-(threshold/max)*H;
        return <g>
          <line x1={padL} x2={width-padR} y1={y} y2={y} stroke="#EF4444" strokeWidth="1" strokeDasharray="4 4" opacity=".7"/>
          <text x={width-padR-4} y={y-4} fontSize="9" textAnchor="end" fill="#EF4444" className="mono">threshold {threshold}</text>
        </g>;
      })()}
      <path d={area} fill={`url(#${id1})`}/>
      <path d={d} fill="none" stroke={`url(#${id2})`} strokeWidth="1.8"/>
      {pts.filter((_,i)=>i%4===0).map((p,i)=>(
        <circle key={i} cx={p[0]} cy={p[1]} r="2" fill={color}/>
      ))}
      {labels && labels.map((l,i)=>(
        i%5===0 && <text key={i} x={padL+i*stepX} y={height-padB+14} fontSize="10" textAnchor="middle" fill="#7F8FAD">{l}</text>
      ))}
    </svg>
  );
}

/* ----------------------- Histogram overlay (failure vs normal) ----------------------- */
function HistogramOverlay({ binsA, binsB, labels, height=200, width=860, colorA="#06D6A0", colorB="#EF4444", nameA="Normal", nameB="Failure" }) {
  const maxV = Math.max(...binsA, ...binsB) * 1.1;
  const padL=42,padR=14,padT=14,padB=30;
  const W=width-padL-padR, H=height-padT-padB;
  const bw = W/binsA.length;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      {[0,.25,.5,.75,1].map((t,i)=>{
        const y=padT+H*(1-t);
        return <line key={i} x1={padL} x2={width-padR} y1={y} y2={y} stroke="rgba(59,130,246,.06)"/>;
      })}
      {binsA.map((v,i)=>{
        const h = (v/maxV)*H;
        return <rect key={'a'+i} x={padL+bw*i+1} y={padT+H-h} width={bw-2} height={h} fill={colorA} opacity=".55" rx="1.5"/>;
      })}
      {binsB.map((v,i)=>{
        const h = (v/maxV)*H;
        return <rect key={'b'+i} x={padL+bw*i+1} y={padT+H-h} width={bw-2} height={h} fill={colorB} opacity=".55" rx="1.5"/>;
      })}
      {labels.map((l,i)=>(
        i%2===0 && <text key={i} x={padL+bw*i+bw/2} y={height-padB+14} fontSize="9" textAnchor="middle" fill="#7F8FAD" className="mono">{l}</text>
      ))}
      {/* legend */}
      <g transform={`translate(${width-180},${padT+4})`}>
        <rect x="0" y="-2" width="170" height="34" rx="6" fill="rgba(10,22,40,.7)" stroke="rgba(59,130,246,.18)"/>
        <rect x="8" y="4" width="10" height="10" rx="2" fill={colorA} opacity=".7"/>
        <text x="24" y="13" fontSize="10" fill="#E8F0FF">{nameA}</text>
        <rect x="8" y="18" width="10" height="10" rx="2" fill={colorB} opacity=".7"/>
        <text x="24" y="27" fontSize="10" fill="#E8F0FF">{nameB}</text>
      </g>
    </svg>
  );
}

/* ----------------------- Gauge (arc 0-100%) ----------------------- */
function ArcGauge({ value=0, label, sub, size=320 }) {
  // value 0..1
  const cx = size/2, cy = size/2+10;
  const r = size/2 - 30;
  const start = Math.PI*0.8, end = Math.PI*2.2; // sweep ~252deg
  const sweep = end-start;
  const ang = start + sweep*clamp(value,0,1);
  // points along arc
  const polar = (angle,rad)=>[cx+Math.cos(angle)*rad, cy+Math.sin(angle)*rad];
  const arcPath = (a0,a1,rad)=>{
    const [x0,y0]=polar(a0,rad), [x1,y1]=polar(a1,rad);
    const large = (a1-a0) > Math.PI ? 1 : 0;
    return `M ${x0} ${y0} A ${rad} ${rad} 0 ${large} 1 ${x1} ${y1}`;
  };
  const id = useMemo(()=>'g-'+Math.random().toString(36).slice(2,7),[]);
  // needle
  const [nx,ny] = polar(ang, r-8);
  const [bx,by] = polar(ang+Math.PI, 14);
  const color = value>0.75 ? '#EF4444' : value>0.5 ? '#F59E0B' : value>0.3 ? '#FBBF24' : '#06D6A0';
  // tick marks
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#06D6A0"/>
          <stop offset=".5" stopColor="#F59E0B"/>
          <stop offset="1" stopColor="#EF4444"/>
        </linearGradient>
        <filter id="glow"><feGaussianBlur stdDeviation="3"/></filter>
      </defs>
      {/* track */}
      <path d={arcPath(start,end,r)} fill="none" stroke="rgba(59,130,246,.12)" strokeWidth="22" strokeLinecap="round"/>
      {/* value */}
      <path d={arcPath(start,ang,r)} fill="none" stroke={`url(#${id})`} strokeWidth="22" strokeLinecap="round" filter="url(#glow)" opacity=".95"/>
      <path d={arcPath(start,ang,r)} fill="none" stroke={`url(#${id})`} strokeWidth="22" strokeLinecap="round"/>
      {/* ticks */}
      {range(11).map(i=>{
        const a = start + sweep*(i/10);
        const [x1,y1]=polar(a,r+12), [x2,y2]=polar(a,r+18);
        return <g key={i}>
          <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#7F8FAD" strokeWidth="1"/>
          <text x={polar(a,r+30)[0]} y={polar(a,r+30)[1]+3} fontSize="9" textAnchor="middle" fill="#7F8FAD" className="mono">{i*10}</text>
        </g>;
      })}
      {/* needle */}
      <line x1={bx} y1={by} x2={nx} y2={ny} stroke="#E8F0FF" strokeWidth="2" strokeLinecap="round"/>
      <circle cx={cx} cy={cy} r="6" fill="#0A1628" stroke={color} strokeWidth="2"/>
      {/* center label */}
      <text x={cx} y={cy-30} textAnchor="middle" fontSize="11" fill="#7F8FAD" letterSpacing=".12em">{sub}</text>
      <text x={cx} y={cy+8} textAnchor="middle" fontSize="48" fontWeight="700" fill={color} className="grotesk tabular" style={{filter:`drop-shadow(0 0 12px ${color}66)`}}>{(value*100).toFixed(1)}%</text>
      <text x={cx} y={cy+28} textAnchor="middle" fontSize="10" fill="#7F8FAD" letterSpacing=".12em">{label}</text>
    </svg>
  );
}

/* ----------------------- Heatmap (Confusion matrix 2x2) ----------------------- */
function ConfusionMatrix({ matrix, size=200, title }) {
  // matrix [[TN, FP],[FN, TP]]
  const max = Math.max(...matrix.flat());
  const cells = [
    { label:'TN', value: matrix[0][0], i:0, j:0 },
    { label:'FP', value: matrix[0][1], i:0, j:1 },
    { label:'FN', value: matrix[1][0], i:1, j:0 },
    { label:'TP', value: matrix[1][1], i:1, j:1 },
  ];
  const cell = size/2;
  return (
    <div style={{display:'flex',flexDirection:'column',gap:8}}>
      {title && <div style={{fontSize:11,color:'#7F8FAD',letterSpacing:'.12em',textTransform:'uppercase'}}>{title}</div>}
      <svg width={size+40} height={size+40} viewBox={`0 0 ${size+40} ${size+40}`}>
        {/* axis labels */}
        <text x="20" y="14" fontSize="9" fill="#7F8FAD" letterSpacing=".1em">PREDICTED →</text>
        <text x="14" y={size/2+30} fontSize="9" fill="#7F8FAD" letterSpacing=".1em" transform={`rotate(-90 14 ${size/2+30})`}>ACTUAL →</text>
        <text x={cell/2+30} y="26" fontSize="9" fill="#B7C5DD" textAnchor="middle">No fail</text>
        <text x={cell + cell/2 + 30} y="26" fontSize="9" fill="#B7C5DD" textAnchor="middle">Failure</text>
        <text x={26} y={cell/2+38} fontSize="9" fill="#B7C5DD" textAnchor="end">No fail</text>
        <text x={26} y={cell + cell/2 + 38} fontSize="9" fill="#B7C5DD" textAnchor="end">Failure</text>
        {cells.map((c,k)=>{
          const t = c.value/max;
          const fill = c.label==='TN' || c.label==='TP'
            ? `rgba(6,214,160,${0.18 + t*0.55})`
            : `rgba(239,68,68,${0.18 + t*0.55})`;
          const stroke = c.label==='TN' || c.label==='TP' ? 'rgba(6,214,160,.5)' : 'rgba(239,68,68,.5)';
          return (
            <g key={k} transform={`translate(${30 + c.j*cell}, ${30 + c.i*cell})`}>
              <rect x="2" y="2" width={cell-4} height={cell-4} rx="6" fill={fill} stroke={stroke}/>
              <text x={cell/2} y={cell/2-4} textAnchor="middle" fontSize="22" fontWeight="700" fill="#E8F0FF" className="grotesk tabular">{c.value.toLocaleString()}</text>
              <text x={cell/2} y={cell/2+14} textAnchor="middle" fontSize="9" fill="#B7C5DD" letterSpacing=".12em">{c.label}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/* ----------------------- Curve (ROC / PR) ----------------------- */
function CurveChart({ curves, xLabel="FPR", yLabel="TPR", height=240, width=380, diagonal=true }) {
  // curves: [{ name, color, points: [[x,y]], auc }]
  const padL=42,padR=16,padT=16,padB=34;
  const W=width-padL-padR, H=height-padT-padB;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      {/* grid */}
      {[0,.25,.5,.75,1].map((t,i)=>(
        <g key={i}>
          <line x1={padL} x2={padL+W} y1={padT+H-t*H} y2={padT+H-t*H} stroke="rgba(59,130,246,.07)"/>
          <line y1={padT} y2={padT+H} x1={padL+t*W} x2={padL+t*W} stroke="rgba(59,130,246,.07)"/>
          <text x={padL-6} y={padT+H-t*H+3} fontSize="9" textAnchor="end" fill="#7F8FAD" className="mono">{t.toFixed(1)}</text>
          <text x={padL+t*W} y={padT+H+14} fontSize="9" textAnchor="middle" fill="#7F8FAD" className="mono">{t.toFixed(1)}</text>
        </g>
      ))}
      {diagonal && (
        <line x1={padL} y1={padT+H} x2={padL+W} y2={padT} stroke="rgba(127,143,173,.45)" strokeDasharray="3 4"/>
      )}
      {curves.map((c,i)=>{
        const d = c.points.map((p,j)=>(j===0?'M':'L')+(padL+p[0]*W).toFixed(1)+' '+(padT+H-p[1]*H).toFixed(1)).join(' ');
        return <path key={i} d={d} fill="none" stroke={c.color} strokeWidth="1.8"/>;
      })}
      {/* axis titles */}
      <text x={padL+W/2} y={height-6} fontSize="10" fill="#7F8FAD" textAnchor="middle">{xLabel}</text>
      <text x="12" y={padT+H/2} fontSize="10" fill="#7F8FAD" textAnchor="middle" transform={`rotate(-90 12 ${padT+H/2})`}>{yLabel}</text>
    </svg>
  );
}

/* ----------------------- Radar (spider) ----------------------- */
function Radar({ axes, series, size=280, max=1 }) {
  // axes: string[], series: [{name, color, values:[]}]
  const cx=size/2,cy=size/2,r=size/2-40;
  const n=axes.length;
  const ang = (i)=> -Math.PI/2 + (i/n)*Math.PI*2;
  const polar=(i,v)=>[cx+Math.cos(ang(i))*(r*v/max), cy+Math.sin(ang(i))*(r*v/max)];
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {[.25,.5,.75,1].map((t,k)=>{
        const pts = range(n).map(i=>{
          const [x,y]=polar(i,t*max);
          return `${x},${y}`;
        }).join(' ');
        return <polygon key={k} points={pts} fill="none" stroke="rgba(59,130,246,.12)"/>;
      })}
      {axes.map((a,i)=>{
        const [x,y]=polar(i,max);
        return <g key={i}>
          <line x1={cx} y1={cy} x2={x} y2={y} stroke="rgba(59,130,246,.1)"/>
          <text x={cx+Math.cos(ang(i))*(r+18)} y={cy+Math.sin(ang(i))*(r+18)+3} fontSize="10" textAnchor="middle" fill="#B7C5DD">{a}</text>
        </g>;
      })}
      {series.map((s,si)=>{
        const pts = s.values.map((v,i)=>polar(i,v)).map(p=>`${p[0]},${p[1]}`).join(' ');
        return <g key={si}>
          <polygon points={pts} fill={s.color} fillOpacity=".12" stroke={s.color} strokeWidth="1.8"/>
          {s.values.map((v,i)=>{
            const [x,y]=polar(i,v);
            return <circle key={i} cx={x} cy={y} r="3" fill={s.color}/>;
          })}
        </g>;
      })}
    </svg>
  );
}

/* ----------------------- Horizontal bar (SHAP) ----------------------- */
function HBarChart({ data, height=380, width=620, barH=20, gap=8, animate=true }) {
  // data: [{label, value, color?}]
  const max = Math.max(...data.map(d=>d.value));
  const padL=190,padR=70,padT=4;
  const W = width - padL - padR;
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      {data.map((d,i)=>{
        const y = padT + i*(barH+gap);
        const w = (d.value/max)*W;
        const t = d.value/max;
        // gradient light->dark blue based on importance
        const col = `oklch(${72 - t*22}% 0.18 ${245 - t*5})`;
        return (
          <g key={i}>
            <text x={padL-8} y={y+barH/2+4} fontSize="11" fill="#B7C5DD" textAnchor="end" className="mono">{d.label}</text>
            <rect x={padL} y={y} width={W} height={barH} rx="4" fill="rgba(59,130,246,.06)"/>
            <rect x={padL} y={y} width={w} height={barH} rx="4" fill={d.color||col}>
              {animate && <animate attributeName="width" from="0" to={w} dur="0.8s" fill="freeze" begin={`${i*0.05}s`}/>}
            </rect>
            <text x={padL+w+8} y={y+barH/2+4} fontSize="10" fill="#E8F0FF" className="mono" fontWeight="600">{d.value.toFixed(3)}</text>
          </g>
        );
      })}
    </svg>
  );
}

Object.assign(window, { Sparkline, BarChart, GroupedBars, Donut, AreaChart, HistogramOverlay, ArcGauge, ConfusionMatrix, CurveChart, Radar, HBarChart, range, lerp, clamp, fmtNum });
