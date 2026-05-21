function PageOverview() {
  const { useState, useEffect } = React;

  const [overview, setOverview] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:8000/data/overview')
      .then(r => r.json())
      .then(d => { setOverview(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const sparks = {
    CNC:        [22,24,21,23,26,28,25,27,24,26,29,31,28,30,32,34,31,33,35],
    Pump:       [48,50,52,49,53,55,57,54,56,58,60,62,65,68,72,75,78,82,85],
    Compressor: [38,36,37,35,36,38,37,39,38,37,36,38,37,38,36,37,38,37,36],
    'Robotic Arm': [62,64,61,63,66,64,67,69,71,68,72,75,73,76,78,75,77,80,82],
  };

  const TYPE_INFO = {
    CNC:         { type:'cnc',        code:'ASSY-CNC · Zone B2',   status:'ok',   sensors:(fr)=>[{label:'VIBRATION',value:'0.42',unit:'m/s²'},{label:'TEMP MOTOR',value:'62.1',unit:'°C'},{label:'CURRENT',value:'8.4',unit:'A'}] },
    Pump:        { type:'pump',       code:'PAINT-PMP · Zone C1',  status:'warn', sensors:(fr)=>[{label:'PRESSURE',value:'28.2',unit:'bar',warn:true},{label:'TEMP MOTOR',value:'81.4',unit:'°C',warn:true},{label:'RPM',value:'2 140',unit:''}] },
    Compressor:  { type:'compressor', code:'BODY-CMP · Zone A4',   status:'ok',   sensors:(fr)=>[{label:'PRESSURE',value:'22.1',unit:'bar'},{label:'TEMP MOTOR',value:'58.2',unit:'°C'},{label:'AMBIENT',value:'13.1',unit:'°C'}] },
    'Robotic Arm':{ type:'robot',     code:'ASSY-RBT · Zone B4',   status:'alert',sensors:(fr)=>[{label:'VIBRATION',value:'4.81',unit:'m/s²',warn:true},{label:'CURRENT',value:'17.4',unit:'A',warn:true},{label:'RPM',value:'2 612',unit:''}] },
  };

  const failureTimeline = [3,5,4,6,5,7,4,5,8,6,7,9,11,8,10,12,9,11,13,10,12,15,11,13,16,12,14,17,13,15];
  const labels = range(30).map(i=>(i+1).toString().padStart(2,'0'));

  const [sensor, setSensor] = useState('vibration_rms');
  const sensorBins = {
    vibration_rms:    { labels:['0.1','0.3','0.5','0.7','0.9','1.1','1.3','1.5','1.7','1.9','2.1','2.3','2.5','2.7','2.9','3.1'], normal:[120,340,820,1450,1820,1620,1180,720,420,210,90,40,20,8,3,1], failure:[2,5,12,28,55,98,165,235,310,365,392,355,290,210,140,75] },
    temperature_motor:{ labels:['40','45','50','55','60','65','70','75','80','85','90','95'], normal:[80,210,520,920,1280,1480,1320,980,640,320,180,80], failure:[1,3,8,22,48,95,168,235,295,338,365,332] },
    current_phase_avg:{ labels:['2','4','6','8','10','12','14','16','18','20'], normal:[60,180,420,720,1100,1480,1620,1280,820,420], failure:[2,5,11,25,52,98,165,238,305,358] },
    pressure_level:   { labels:['5','10','15','20','25','30','35','40','45','50'], normal:[40,130,310,580,920,1320,1580,1480,1080,640], failure:[3,7,15,32,62,108,178,245,315,365] },
    rpm:              { labels:['500','1000','1500','2000','2500','3000'], normal:[120,420,980,1620,1080,420], failure:[2,12,78,245,365,288] },
  };
  const sd = sensorBins[sensor];

  const fr = overview?.failure_rate ?? 0;
  const byType = overview?.by_machine_type ?? {};
  const machineTypes = Object.keys(TYPE_INFO);

  const barData = machineTypes.map(t => (byType[t]?.failure_rate ?? 0) * 100);
  const failuresCount = overview?.failures_count ?? 0;
  const avgRul = overview?.avg_rul ?? 0;
  const totalMachines = overview?.total_machines ?? 0;

  return (
    <div style={{display:'flex',flexDirection:'column',gap:22,padding:'24px 28px 40px'}}>

      {/* KPI row */}
      <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:16}}>
        <KPICard accent="red"
          label="Failure rate · 24h window"
          value={loading ? '…' : (fr*100).toFixed(1)}
          unit="%"
          delta={2.3} deltaLabel="vs 7d"
          icon={<I.AlertTriangle size={16}/>}
          sub={loading ? 'loading…' : `${failuresCount.toLocaleString()} failures across ${(overview?.total_records??0).toLocaleString()} obs.`}
        />
        <KPICard accent="blue"
          label="Machine types monitored"
          value="4"
          sub="CNC · Pump · Compressor · Robotic Arm"
          icon={<I.Cpu size={16}/>}
          valueColor="#E8F0FF"
        />
        <KPICard accent="green"
          label="Average RUL"
          value={loading ? '…' : Math.round(avgRul)}
          unit="h"
          delta={4.1} deltaLabel="vs 7d"
          icon={<I.Activity size={16}/>}
          sub={loading ? 'loading…' : `≈ ${(avgRul/24).toFixed(1)} days mean residual life`}
          valueColor="#06D6A0"
        />
        <KPICard accent="amber"
          label="Machines monitored"
          value={loading ? '…' : totalMachines}
          sub={loading ? 'loading…' : `across ${machineTypes.length} machine types`}
          icon={<I.Wrench size={16}/>}
          valueColor="#F59E0B"
        />
      </div>

      {/* Machine grid */}
      <div>
        <SectionTitle accent="#3B82F6" action={
          <div style={{display:'flex',gap:8}}>
            <Pill active>All zones</Pill><Pill>Body line</Pill><Pill>Paint shop</Pill><Pill>Assembly</Pill>
          </div>
        }>Fleet status · {machineTypes.length} machine types</SectionTitle>
        <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:14}}>
          {machineTypes.map((mt, i) => {
            const info = TYPE_INFO[mt];
            const mfr = byType[mt]?.failure_rate ?? 0;
            const mrul = byType[mt]?.avg_rul ?? 0;
            const mcnt = byType[mt]?.machine_count ?? 0;
            return (
              <MachineCard key={mt}
                name={`${mt} Fleet`}
                code={`${mcnt} unit${mcnt>1?'s':''} · ${info.code}`}
                type={info.type}
                status={mfr > 0.2 ? 'alert' : mfr > 0.12 ? 'warn' : 'ok'}
                spark={sparks[mt]}
                last={{ rul: Math.round(mrul), prob: (mfr*100).toFixed(1) }}
                sensors={info.sensors(mfr)}
              />
            );
          })}
        </div>
      </div>

      {/* Mid charts */}
      <div style={{display:'grid',gridTemplateColumns:'1.1fr 1fr 1fr',gap:16}}>
        <div className="glass" style={{padding:18,borderRadius:14}}>
          <SectionTitle accent="#3B82F6">Failure rate by machine type · real data</SectionTitle>
          <BarChart
            data={barData}
            labels={machineTypes.map(t => t === 'Robotic Arm' ? 'Robot' : t)}
            color="#3B82F6"
            width={420} height={210}
            format={v=>v.toFixed(1)+'%'}
            maxValue={Math.max(...barData, 25)}
          />
          <div style={{fontSize:11,color:'#7F8FAD',marginTop:4,display:'flex',justifyContent:'space-between'}}>
            <span>n={overview?.total_records?.toLocaleString()??'…'} observations</span>
            <span className="mono" style={{color:'#06D6A0'}}>fleet avg {(fr*100).toFixed(1)}%</span>
          </div>
        </div>

        <div className="glass" style={{padding:18,borderRadius:14}}>
          <SectionTitle accent="#00B4D8">Failure mode distribution</SectionTitle>
          <Donut
            size={170} thickness={20}
            centerLabel={failuresCount>0?failuresCount.toLocaleString():'…'}
            centerSub="FAILURES · DATASET"
            slices={[
              { label:'Bearing wear',   value:31, color:'#3B82F6' },
              { label:'Motor overheat', value:28, color:'#F59E0B' },
              { label:'Hydraulic',      value:22, color:'#00B4D8' },
              { label:'Electrical',     value:19, color:'#8B5CF6' },
            ]}
          />
        </div>

        <div className="glass" style={{padding:18,borderRadius:14}}>
          <SectionTitle accent="#06D6A0" action={<span className="mono" style={{fontSize:10.5,color:'#7F8FAD'}}>by ML class</span>}>Failure rate per machine type</SectionTitle>
          <div style={{display:'flex',flexDirection:'column',gap:10,marginTop:4}}>
            {machineTypes.map((mt,i)=>{
              const mfr = byType[mt]?.failure_rate ?? 0;
              const colors = ['#3B82F6','#00B4D8','#8B5CF6','#F59E0B'];
              return (
                <div key={i}>
                  <div style={{display:'flex',justifyContent:'space-between',fontSize:11.5,marginBottom:4}}>
                    <span style={{display:'flex',alignItems:'center',gap:6}}><span style={{width:8,height:8,borderRadius:2,background:colors[i]}}/>{mt}</span>
                    <span className="mono" style={{color:'#7F8FAD'}}>{(mfr*100).toFixed(2)}%</span>
                  </div>
                  <div style={{display:'flex',height:6,borderRadius:3,overflow:'hidden',background:'rgba(10,22,40,.6)'}}>
                    <div style={{flex:mfr,background:colors[i],boxShadow:`0 0 8px ${colors[i]}66`}}/>
                    <div style={{flex:1-mfr,background:'transparent'}}/>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Timeline */}
      <div className="glass" style={{padding:18,borderRadius:14}}>
        <SectionTitle accent="#3B82F6" action={
          <div style={{display:'flex',gap:8}}>
            <Pill>24h</Pill><Pill>7d</Pill><Pill active>30d</Pill><Pill>90d</Pill>
            <button style={{display:'flex',alignItems:'center',gap:6,padding:'6px 10px',background:'rgba(10,22,40,.5)',border:'1px solid rgba(59,130,246,.18)',borderRadius:8,color:'#B7C5DD',fontSize:11,cursor:'pointer'}}><I.Download size={12}/> Export CSV</button>
          </div>
        }>Failure alerts over time</SectionTitle>
        <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:8}}>
          <div><span className="grotesk tabular" style={{fontSize:26,fontWeight:700,color:'#E8F0FF',letterSpacing:'-.02em'}}>284</span><span style={{fontSize:11.5,color:'#7F8FAD',marginLeft:8}}>predicted alerts · 30d</span></div>
          <div style={{padding:'4px 8px',background:'rgba(239,68,68,.1)',border:'1px solid rgba(239,68,68,.3)',borderRadius:6,color:'#EF4444',fontSize:11,fontWeight:600}}>+18.4% vs prev. 30d</div>
          <div style={{flex:1}}/>
          <div style={{display:'flex',gap:14,fontSize:11,color:'#7F8FAD'}}>
            <span><span style={{width:10,height:2,background:'#3B82F6',display:'inline-block',marginRight:5}}/>Predicted alerts</span>
          </div>
        </div>
        <AreaChart data={failureTimeline} labels={labels} color="#3B82F6" color2="#00B4D8" width={1080} height={210} threshold={14}/>
      </div>

      {/* Sensor explorer */}
      <div className="glass" style={{padding:18,borderRadius:14}}>
        <SectionTitle accent="#8B5CF6" action={
          <div className="mono" style={{fontSize:10.5,color:'#7F8FAD'}}>distribution overlay · failure vs normal</div>
        }>Sensor explorer</SectionTitle>
        <div style={{display:'flex',gap:8,marginBottom:14}}>
          {Object.keys(sensorBins).map(k=>(
            <Pill key={k} active={sensor===k} onClick={()=>setSensor(k)}>{k}</Pill>
          ))}
        </div>
        <div style={{display:'grid',gridTemplateColumns:'1fr 240px',gap:16,alignItems:'center'}}>
          <HistogramOverlay binsA={sd.normal} binsB={sd.failure} labels={sd.labels} width={840} height={220}/>
          <div style={{display:'flex',flexDirection:'column',gap:10}}>
            <div style={{padding:12,background:'rgba(10,22,40,.5)',borderRadius:10,border:'1px solid rgba(6,214,160,.18)'}}>
              <div style={{fontSize:10,color:'#7F8FAD',letterSpacing:'.12em'}}>NORMAL · μ ± σ</div>
              <div className="grotesk tabular" style={{fontSize:18,fontWeight:700,color:'#06D6A0',marginTop:3}}>{sensor==='temperature_motor'?'58.2 ± 6.4 °C':sensor==='vibration_rms'?'0.68 ± 0.34 m/s²':sensor==='current_phase_avg'?'7.4 ± 3.1 A':sensor==='pressure_level'?'21.0 ± 4.2 bar':'1 240 ± 380 rpm'}</div>
            </div>
            <div style={{padding:12,background:'rgba(10,22,40,.5)',borderRadius:10,border:'1px solid rgba(239,68,68,.18)'}}>
              <div style={{fontSize:10,color:'#7F8FAD',letterSpacing:'.12em'}}>FAILURE · μ ± σ</div>
              <div className="grotesk tabular" style={{fontSize:18,fontWeight:700,color:'#EF4444',marginTop:3}}>{sensor==='temperature_motor'?'86.6 ± 7.2 °C':sensor==='vibration_rms'?'4.16 ± 1.68 m/s²':sensor==='current_phase_avg'?'15.8 ± 2.2 A':sensor==='pressure_level'?'38.2 ± 5.1 bar':'2 480 ± 320 rpm'}</div>
            </div>
            <div style={{padding:12,background:'rgba(59,130,246,.08)',borderRadius:10,border:'1px solid rgba(59,130,246,.22)'}}>
              <div style={{fontSize:10,color:'#7F8FAD',letterSpacing:'.12em'}}>SHAP IMPORTANCE</div>
              <div className="grotesk tabular" style={{fontSize:18,fontWeight:700,color:'#3B82F6',marginTop:3}}>{sensor==='rpm'?'1.664':sensor==='temperature_motor'?'1.556':sensor==='vibration_rms'?'0.932':sensor==='current_phase_avg'?'0.813':'0.492'}</div>
              <div style={{fontSize:10.5,color:'#B7C5DD',marginTop:3}}>mean |SHAP| · XGBoost model</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

window.PageOverview = PageOverview;
