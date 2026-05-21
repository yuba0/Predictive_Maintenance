const { useState, useEffect } = React;

function App() {
  const [showIntro, setShowIntro] = useState(true);
  const [page, setPage] = useState(()=> location.hash.replace('#','') || 'overview');
  useEffect(()=>{
    const h = ()=> setPage(location.hash.replace('#','') || 'overview');
    window.addEventListener('hashchange', h);
    return ()=> window.removeEventListener('hashchange', h);
  },[]);
  const onNav = (id)=>{ location.hash = id; setPage(id); window.scrollTo({top:0,behavior:'smooth'}); };

  const meta = {
    overview:   { title:'Live Overview', sub:'Real-time fleet health · 24h horizon', crumb:'WORKSPACE  /  LIVE OVERVIEW' },
    simulator:  { title:'Prediction Simulator', sub:'Synthetic inference console', crumb:'WORKSPACE  /  SIMULATOR' },
    models:     { title:'Model Performance', sub:'4 classifiers benchmarked', crumb:'WORKSPACE  /  MODEL PERFORMANCE' },
    shap:       { title:'SHAP Interpretability', sub:'Why the model predicts what it predicts', crumb:'WORKSPACE  /  INTERPRETABILITY' },
  };
  const m = meta[page] || meta.overview;

  let body = null;
  if (page==='overview')  body = <PageOverview/>;
  else if (page==='simulator') body = <PageSimulator/>;
  else if (page==='models') body = <PageModels/>;
  else if (page==='shap') body = <PageShap/>;
  else body = <PageOverview/>;

  return (
    <div style={{display:'flex',minHeight:'100vh',position:'relative'}}>
      {showIntro && <IntroScreen onComplete={() => setShowIntro(false)}/>}
      <Sidebar active={page} onNav={onNav}/>
      <main style={{flex:1,minWidth:0,display:'flex',flexDirection:'column'}} data-screen-label={page}>
        <Topbar title={m.title} sub={m.sub} breadcrumb={m.crumb}/>
        <div className="fade-up" key={page}>{body}</div>
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
