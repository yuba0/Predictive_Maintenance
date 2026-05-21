const { useEffect, useRef, useState, useCallback } = React;

function IntroScreen({ onComplete }) {
  const canvasRef = useRef(null);
  const [phase, setPhase] = useState('visible');
  const dismissedRef = useRef(false);

  const dismiss = useCallback(() => {
    if (dismissedRef.current) return;
    dismissedRef.current = true;
    setPhase('fading');
    setTimeout(() => onComplete && onComplete(), 500);
  }, [onComplete]);

  useEffect(() => {
    const t = setTimeout(dismiss, 2800);
    return () => clearTimeout(t);
  }, [dismiss]);

  useEffect(() => {
    const THREE = window.THREE;
    if (!THREE || !canvasRef.current) return;
    const canvas = canvasRef.current;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(420, 420);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    camera.position.z = 6.5;

    // Lighting
    scene.add(new THREE.AmbientLight(0x0d1f3d, 3));
    const key = new THREE.DirectionalLight(0x7ab3f7, 5);
    key.position.set(4, 5, 6);
    scene.add(key);
    const fill = new THREE.PointLight(0x00B4D8, 4, 22);
    fill.position.set(-5, -2, 3);
    scene.add(fill);
    const rim = new THREE.PointLight(0x06D6A0, 2.5, 15);
    rim.position.set(2, 5, -4);
    scene.add(rim);

    // --- Coin group ---
    const coinGroup = new THREE.Group();
    scene.add(coinGroup);

    const darkMat  = new THREE.MeshStandardMaterial({ color: 0x0F1D33, metalness: 0.96, roughness: 0.08 });
    const blueMat  = new THREE.MeshStandardMaterial({ color: 0x2563EB, metalness: 0.75, roughness: 0.18, emissive: new THREE.Color(0x0a1f55), emissiveIntensity: 0.5 });
    const cyanMat  = new THREE.MeshStandardMaterial({ color: 0x00B4D8, metalness: 0.85, roughness: 0.1,  emissive: new THREE.Color(0x002233), emissiveIntensity: 0.6 });
    const edgeMat  = new THREE.MeshStandardMaterial({ color: 0x1B345A, metalness: 0.98, roughness: 0.05 });

    // Main coin body (CylinderGeometry along Y → rotate to face camera via X rotation)
    const body = new THREE.Mesh(new THREE.CylinderGeometry(1.5, 1.5, 0.28, 80), darkMat);
    body.rotation.x = Math.PI / 2;
    coinGroup.add(body);

    // Front face disc
    const frontFace = new THREE.Mesh(new THREE.CylinderGeometry(1.4, 1.4, 0.012, 80), blueMat);
    frontFace.rotation.x = Math.PI / 2;
    frontFace.position.z = 0.147;
    coinGroup.add(frontFace);

    // Outer ring on face
    const outerRing = new THREE.Mesh(new THREE.TorusGeometry(1.05, 0.026, 16, 80), cyanMat);
    outerRing.position.z = 0.15;
    coinGroup.add(outerRing);

    // Inner ring on face
    const innerRing = new THREE.Mesh(new THREE.TorusGeometry(0.52, 0.018, 16, 60), cyanMat.clone());
    innerRing.position.z = 0.15;
    coinGroup.add(innerRing);

    // Center emblem
    const emblem = new THREE.Mesh(new THREE.SphereGeometry(0.19, 32, 32), blueMat.clone());
    emblem.position.z = 0.16;
    coinGroup.add(emblem);

    // 3 spokes spanning inner to outer ring
    // CylinderGeometry axis is Y; rotation.z = a - PI/2 points it along radial direction a
    for (let i = 0; i < 3; i++) {
      const a = (i / 3) * Math.PI * 2;
      const spoke = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 1.05, 8), cyanMat.clone());
      spoke.position.z = 0.15;
      spoke.rotation.z = a - Math.PI / 2;
      coinGroup.add(spoke);
    }

    // Gear teeth (22 teeth around perimeter)
    for (let i = 0; i < 22; i++) {
      const a = (i / 22) * Math.PI * 2;
      const tooth = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.13, 0.3), edgeMat.clone());
      tooth.position.set(Math.cos(a) * 1.63, Math.sin(a) * 1.63, 0);
      tooth.rotation.z = a;
      coinGroup.add(tooth);
    }

    // --- Floating particles ---
    const pN = 90;
    const pPositions = new Float32Array(pN * 3);
    const pVelocities = new Float32Array(pN * 3);
    for (let i = 0; i < pN; i++) {
      pPositions[i * 3]     = (Math.random() - 0.5) * 14;
      pPositions[i * 3 + 1] = (Math.random() - 0.5) * 9;
      pPositions[i * 3 + 2] = (Math.random() - 0.5) * 5 - 2;
      pVelocities[i * 3]     = (Math.random() - 0.5) * 0.005;
      pVelocities[i * 3 + 1] = (Math.random() - 0.5) * 0.005;
    }
    const pGeo = new THREE.BufferGeometry();
    const pAttr = new THREE.BufferAttribute(pPositions, 3);
    pGeo.setAttribute('position', pAttr);
    const particles = new THREE.Points(pGeo, new THREE.PointsMaterial({ color: 0x3B82F6, size: 0.042, transparent: true, opacity: 0.5 }));
    scene.add(particles);

    let raf;
    let t = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      t += 0.016;

      coinGroup.rotation.y += 0.024;
      coinGroup.rotation.x = Math.sin(t * 0.38) * 0.11;

      // Drift particles and wrap at boundaries
      const pos = pAttr.array;
      for (let i = 0; i < pN; i++) {
        pos[i * 3]     += pVelocities[i * 3];
        pos[i * 3 + 1] += pVelocities[i * 3 + 1];
        if (pos[i * 3]     >  7) pos[i * 3]     = -7;
        if (pos[i * 3]     < -7) pos[i * 3]     =  7;
        if (pos[i * 3 + 1] >  4.5) pos[i * 3 + 1] = -4.5;
        if (pos[i * 3 + 1] < -4.5) pos[i * 3 + 1] =  4.5;
      }
      pAttr.needsUpdate = true;

      renderer.render(scene, camera);
    };
    tick();

    return () => {
      cancelAnimationFrame(raf);
      renderer.dispose();
    };
  }, []);

  return (
    <div
      onClick={dismiss}
      style={{
        position: 'fixed', inset: 0, zIndex: 9999,
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        background: 'radial-gradient(ellipse 130% 70% at 50% 0%, rgba(59,130,246,.22) 0%, transparent 58%), #070F1F',
        cursor: 'pointer', userSelect: 'none',
        opacity: phase === 'fading' ? 0 : 1,
        transition: 'opacity 0.5s ease',
      }}
    >
      {/* Grid texture */}
      <div style={{
        position: 'absolute', inset: 0, pointerEvents: 'none',
        backgroundImage: 'linear-gradient(rgba(59,130,246,.06) 1px,transparent 1px),linear-gradient(90deg,rgba(59,130,246,.06) 1px,transparent 1px)',
        backgroundSize: '48px 48px',
        maskImage: 'radial-gradient(ellipse 75% 75% at 50% 50%, black 0%, transparent 72%)',
        WebkitMaskImage: 'radial-gradient(ellipse 75% 75% at 50% 50%, black 0%, transparent 72%)',
      }}/>

      {/* Blue halo behind coin */}
      <div style={{
        position: 'absolute', width: 400, height: 400, borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(59,130,246,.24) 0%, transparent 65%)',
        filter: 'blur(28px)', pointerEvents: 'none',
      }}/>

      {/* Three.js canvas — sized by renderer.setSize(420, 420) */}
      <canvas
        ref={canvasRef}
        style={{ display: 'block', filter: 'drop-shadow(0 0 52px rgba(59,130,246,.9))' }}
      />

      {/* Title */}
      <div style={{ marginTop: 18, textAlign: 'center', animation: 'fadeUp .7s .1s ease-out both' }}>
        <div style={{
          fontFamily: "'Space Grotesk', sans-serif",
          fontSize: 42, fontWeight: 700, letterSpacing: '-0.025em', color: '#E8F0FF', lineHeight: 1,
        }}>
          <span style={{ color: '#3B82F6' }}>Predict</span>Maint
        </div>
        <div style={{
          marginTop: 9, fontSize: 11, color: '#7F8FAD',
          letterSpacing: '0.26em', textTransform: 'uppercase',
          fontFamily: "'JetBrains Mono', monospace",
        }}>
          Predictive Maintenance Platform
        </div>
      </div>

      {/* Progress bar */}
      <div style={{
        marginTop: 28, width: 200, height: 2,
        background: 'rgba(59,130,246,.15)', borderRadius: 99, overflow: 'hidden',
        animation: 'fadeUp .4s .4s ease-out both',
      }}>
        <div style={{
          height: '100%',
          background: 'linear-gradient(90deg, #3B82F6, #00B4D8, #06D6A0)',
          borderRadius: 99,
          animation: 'introBar 2.6s cubic-bezier(.4,0,.2,1) forwards',
        }}/>
      </div>

      {/* Tap hint */}
      <div style={{
        marginTop: 14, fontSize: 11, color: '#3d4f6b',
        letterSpacing: '0.14em', textTransform: 'uppercase',
        animation: 'fadeUp .4s .8s ease-out both',
      }}>
        tap to continue
      </div>
    </div>
  );
}

window.IntroScreen = IntroScreen;
