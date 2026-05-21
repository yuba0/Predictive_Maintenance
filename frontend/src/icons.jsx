// Lightweight inline SVG icons. Stroke-based, 1.6 weight.
const Icon = ({ d, size = 18, stroke = "currentColor", fill = "none", strokeWidth = 1.6, viewBox = "0 0 24 24", children, ...rest }) => (
  <svg width={size} height={size} viewBox={viewBox} fill={fill} stroke={stroke} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" {...rest}>
    {d ? <path d={d}/> : children}
  </svg>
);

const I = {
  Logo: (p)=>(
    <svg width={p.size||24} height={p.size||24} viewBox="0 0 32 32" fill="none" {...p}>
      <defs>
        <linearGradient id="lg-logo" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#3B82F6"/>
          <stop offset="1" stopColor="#00B4D8"/>
        </linearGradient>
      </defs>
      <path d="M16 2 L28 9 V23 L16 30 L4 23 V9 Z" stroke="url(#lg-logo)" strokeWidth="1.6" fill="rgba(59,130,246,.08)"/>
      <path d="M10 16 L14 20 L22 12" stroke="url(#lg-logo)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
      <circle cx="16" cy="16" r="9.5" stroke="url(#lg-logo)" strokeWidth=".5" strokeDasharray="2 3" fill="none"/>
    </svg>
  ),
  Overview: (p)=> <Icon {...p}><rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/></Icon>,
  Simulator: (p)=> <Icon {...p}><path d="M12 2v4"/><path d="M12 18v4"/><path d="M4.93 4.93l2.83 2.83"/><path d="M16.24 16.24l2.83 2.83"/><path d="M2 12h4"/><path d="M18 12h4"/><circle cx="12" cy="12" r="4"/></Icon>,
  Models: (p)=> <Icon {...p}><circle cx="6" cy="6" r="2.2"/><circle cx="18" cy="6" r="2.2"/><circle cx="6" cy="18" r="2.2"/><circle cx="18" cy="18" r="2.2"/><circle cx="12" cy="12" r="2.4"/><path d="M8 6.8L10 11.2"/><path d="M16 6.8L14 11.2"/><path d="M8 17.2L10 12.8"/><path d="M16 17.2L14 12.8"/></Icon>,
  Shap: (p)=> <Icon {...p}><path d="M3 3v18h18"/><path d="M7 14l3-4 3 2 5-7"/><circle cx="7" cy="14" r="1.3" fill="currentColor"/><circle cx="10" cy="10" r="1.3" fill="currentColor"/><circle cx="13" cy="12" r="1.3" fill="currentColor"/><circle cx="18" cy="5" r="1.3" fill="currentColor"/></Icon>,
  Bell: (p)=> <Icon {...p}><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a2 2 0 0 0 3.4 0"/></Icon>,
  Search: (p)=> <Icon {...p}><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></Icon>,
  Settings: (p)=> <Icon {...p}><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.8L4.2 7a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></Icon>,
  Bolt: (p)=> <Icon {...p}><path d="M13 2L4 14h7l-1 8 9-12h-7l1-8z" fill={p.fill||'rgba(59,130,246,.18)'}/></Icon>,
  ChevronLeft: (p)=> <Icon {...p}><polyline points="15 18 9 12 15 6"/></Icon>,
  ChevronRight: (p)=> <Icon {...p}><polyline points="9 18 15 12 9 6"/></Icon>,
  ChevronDown: (p)=> <Icon {...p}><polyline points="6 9 12 15 18 9"/></Icon>,
  Plus: (p)=> <Icon {...p}><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></Icon>,
  Download: (p)=> <Icon {...p}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></Icon>,
  Filter: (p)=> <Icon {...p}><polygon points="22 3 2 3 10 12.5 10 19 14 21 14 12.5 22 3"/></Icon>,
  Calendar: (p)=> <Icon {...p}><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></Icon>,
  AlertTriangle: (p)=> <Icon {...p}><path d="M10.3 3.86 1.82 18a2 2 0 0 0 1.7 3h16.96a2 2 0 0 0 1.7-3L13.7 3.86a2 2 0 0 0-3.4 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></Icon>,
  Check: (p)=> <Icon {...p}><polyline points="20 6 9 17 4 12"/></Icon>,
  X: (p)=> <Icon {...p}><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></Icon>,
  TrendUp: (p)=> <Icon {...p}><polyline points="3 17 9 11 13 15 21 7"/><polyline points="14 7 21 7 21 14"/></Icon>,
  TrendDown: (p)=> <Icon {...p}><polyline points="3 7 9 13 13 9 21 17"/><polyline points="14 17 21 17 21 10"/></Icon>,
  Info: (p)=> <Icon {...p}><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></Icon>,
  Cpu: (p)=> <Icon {...p}><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></Icon>,
  Wrench: (p)=> <Icon {...p}><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></Icon>,
  Activity: (p)=> <Icon {...p}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></Icon>,
  Play: (p)=> <Icon {...p}><polygon points="6 4 20 12 6 20 6 4" fill="currentColor" stroke="none"/></Icon>,
  Refresh: (p)=> <Icon {...p}><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10"/><path d="M20.49 15a9 9 0 0 1-14.85 3.36L1 14"/></Icon>,
  Medal1: ()=> <span style={{fontSize:18,filter:'drop-shadow(0 0 6px rgba(245,158,11,.45))'}}>🥇</span>,
  Medal2: ()=> <span style={{fontSize:18,filter:'drop-shadow(0 0 6px rgba(180,200,220,.45))'}}>🥈</span>,
  Medal3: ()=> <span style={{fontSize:18,filter:'drop-shadow(0 0 6px rgba(205,127,50,.45))'}}>🥉</span>,
};

// Machine glyphs — abstract, monoline
const MachineGlyph = {
  cnc: (p) => (
    <svg viewBox="0 0 64 64" width={p.size||44} height={p.size||44} fill="none" stroke={p.color||"currentColor"} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <rect x="6" y="40" width="52" height="14" rx="2"/>
      <rect x="14" y="12" width="36" height="6" rx="1"/>
      <path d="M22 18v10"/><path d="M42 18v10"/>
      <rect x="22" y="28" width="20" height="6" rx="1"/>
      <path d="M28 34v6"/><path d="M36 34v6"/>
      <circle cx="32" cy="47" r="2.5"/>
      <path d="M10 40v-6h8"/>
    </svg>
  ),
  pump: (p) => (
    <svg viewBox="0 0 64 64" width={p.size||44} height={p.size||44} fill="none" stroke={p.color||"currentColor"} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="32" cy="32" r="14"/>
      <path d="M32 22v20M22 32h20" />
      <path d="M24 24l16 16M40 24L24 40" />
      <path d="M46 32h10v6h-8"/>
      <path d="M8 32h10"/>
      <rect x="28" y="50" width="8" height="6" rx="1"/>
    </svg>
  ),
  compressor: (p) => (
    <svg viewBox="0 0 64 64" width={p.size||44} height={p.size||44} fill="none" stroke={p.color||"currentColor"} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <rect x="10" y="18" width="44" height="22" rx="3"/>
      <circle cx="22" cy="29" r="5"/>
      <circle cx="42" cy="29" r="5"/>
      <path d="M22 26v6M42 26v6"/>
      <rect x="18" y="40" width="28" height="10" rx="1"/>
      <path d="M14 50v4M50 50v4"/>
    </svg>
  ),
  robot: (p) => (
    <svg viewBox="0 0 64 64" width={p.size||44} height={p.size||44} fill="none" stroke={p.color||"currentColor"} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <rect x="20" y="48" width="24" height="8" rx="1"/>
      <circle cx="32" cy="46" r="3"/>
      <path d="M32 43L18 22"/>
      <circle cx="18" cy="22" r="3"/>
      <path d="M20 20L40 12"/>
      <circle cx="42" cy="11" r="3"/>
      <path d="M44 12l8 4"/>
      <path d="M52 16l-2 6"/>
    </svg>
  ),
};

Object.assign(window, { I, Icon, MachineGlyph });
