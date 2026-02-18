import React, { useState, useRef, useEffect } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Play, Activity, Database, Terminal } from 'lucide-react';

// --- CONFIGURATION ---
const API_URL = "https://leading-cloth-switched-reseller.trycloudflare.com/v1/completions"; 

export default function RedStringApp() {
  const [inputText, setInputText] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [logs, setLogs] = useState([]);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  
  // Interaction State
  const [hoverLink, setHoverLink] = useState(null);
  const [hoverNode, setHoverNode] = useState(null);
  
  // Refs
  const graphWrapperRef = useRef(null);
  const fgRef = useRef(); 
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  const addLog = (msg) => setLogs(prev => [msg, ...prev]);

  // --- 1. RESIZE OBSERVER ---
  useEffect(() => {
    const resizeObserver = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });
    if (graphWrapperRef.current) resizeObserver.observe(graphWrapperRef.current);
    return () => resizeObserver.disconnect();
  }, []);

  // --- 2. PHYSICS TUNING (Preserved Settings) ---
  useEffect(() => {
    const timeout = setTimeout(() => {
      if (fgRef.current) {
        const chargeForce = fgRef.current.d3Force('charge');
        if (chargeForce) {
          chargeForce.strength(-60); 
          chargeForce.distanceMax(120); 
        }
        fgRef.current.d3Force('center').strength(0.20);
      }
    }, 100);
    return () => clearTimeout(timeout);
  }, []);

  // --- 3. JSON EXTRACTION ---
  const extractJSON = (text) => {
    try {
      const start = text.indexOf('[');
      const end = text.lastIndexOf(']');
      if (start === -1 || end === -1) return null;
      return JSON.parse(text.substring(start, end + 1));
    } catch (e) {
      return null;
    }
  };

  const updateGraph = (triples) => {
    if (!triples || triples.length === 0) return;

    setGraphData(prev => {
      const newNodes = [...prev.nodes];
      const newLinks = [...prev.links];
      let addedCount = 0;

      triples.forEach(t => {
        const headId = t.head?.toLowerCase().trim();
        const tailId = t.tail?.toLowerCase().trim();
        const type = t.type?.toLowerCase().trim();
        if (!headId || !tailId) return;

        if (!newNodes.find(n => n.id === headId)) {
          newNodes.push({ id: headId, label: t.head, group: 1 });
        }
        if (!newNodes.find(n => n.id === tailId)) {
          newNodes.push({ id: tailId, label: t.tail, group: 2 });
        }

        const exists = newLinks.some(l => 
          (l.source.id === headId || l.source === headId) && 
          (l.target.id === tailId || l.target === tailId) &&
          l.type === type
        );
        
        if (!exists) {
          newLinks.push({ source: headId, target: tailId, type: type, label: type });
          addedCount++;
        }
      });

      if (addedCount > 0) addLog(`✨ Added ${addedCount} threads.`);
      return { nodes: newNodes, links: newLinks };
    });
  };

  // --- 4. INVESTIGATION LOGIC ---
  const startInvestigation = async () => {
    if (!inputText) return;
    setIsProcessing(true);
    addLog("🚀 Starting Investigation...");

    const segmenter = new Intl.Segmenter('en', { granularity: 'sentence' });
    const segments = Array.from(segmenter.segment(inputText));
    const sentences = segments.map(s => s.segment.trim()).filter(s => s.length > 0);

    const windows = [];
    for (let i = 0; i < sentences.length; i++) {
      const current = sentences[i];
      const next = sentences[i + 1] || ""; 
      windows.push(`${current} ${next}`.trim());
    }

    addLog(`Text chunked into ${windows.length} segments.`);

    for (let i = 0; i < windows.length; i++) {
      const windowText = windows[i];
      addLog(`🔍 Scanning Window ${i+1}/${windows.length}...`);

      try {
        const payload = {
          prompt: `### Instruction:\nExtract relationship triples as a JSON list.\n\n### Input:\n${windowText}\n\n### Response:\n`,
          max_tokens: 512,
          stop: ["###"]
        };

        const response = await fetch(API_URL, {
          method: "POST",
          headers: { 
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true" 
          },
          body: JSON.stringify(payload)
        });

        const data = await response.json();
        const rawText = data.choices[0].message ? data.choices[0].message.content : data.choices[0].text;
        const triples = extractJSON(rawText);
        
        if (triples && triples.length > 0) updateGraph(triples);

      } catch (err) {
        addLog(`❌ Error: ${err.message}`);
      }
    }
    setIsProcessing(false);
    addLog("✅ Investigation Complete.");
  };

  // --- 5. RENDER ---
  return (
    <>
      <style>
        {`
          body, html, #root { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #0f172a; }
        `}
      </style>

      <div style={{ display: 'flex', width: '100vw', height: '100vh', background: '#0f172a', color: '#f8fafc', fontFamily: 'sans-serif' }}>
        
        {/* LEFT PANEL */}
        <div style={{ width: '400px', borderRight: '1px solid #334155', display: 'flex', flexDirection: 'column', padding: '20px', backgroundColor: '#1e293b' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
            <Activity color="#ef4444" />
            <h1 style={{ fontSize: '1.2rem', margin: 0, fontWeight: 'bold' }}>Red String v2.6</h1>
          </div>

          <textarea 
            placeholder="Paste text here..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            style={{ flexGrow: 1, background: '#0f172a', border: '1px solid #334155', color: '#e2e8f0', padding: '15px', borderRadius: '8px', resize: 'none', fontFamily: 'monospace' }}
          />

          <button 
            onClick={startInvestigation}
            disabled={isProcessing}
            style={{ marginTop: '20px', padding: '12px', background: isProcessing ? '#64748b' : '#ef4444', border: 'none', color: 'white', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', fontWeight: 'bold' }}
          >
            <Play size={18} /> {isProcessing ? "Analyzing..." : "Start Investigation"}
          </button>

          <div style={{ marginTop: '20px', height: '200px', background: '#020617', padding: '10px', borderRadius: '6px', fontSize: '0.8rem', overflowY: 'auto', border: '1px solid #334155' }}>
            {logs.map((log, i) => <div key={i} style={{ marginBottom: '4px', fontFamily: 'monospace', color: '#94a3b8' }}>{log}</div>)}
          </div>
        </div>

        {/* RIGHT PANEL: THE BOARD */}
        <div 
          ref={graphWrapperRef} 
          style={{ flexGrow: 1, position: 'relative', backgroundColor: '#020617', overflow: 'hidden' }}
        >
          <ForceGraph2D
            ref={fgRef}
            width={dimensions.width}
            height={dimensions.height}
            graphData={graphData}
            backgroundColor="#020617"
            
            // --- PHYSICS ---
            d3VelocityDecay={0.5} 
            cooldownTicks={150}   
            
            // --- INTERACTION ---
            onLinkHover={setHoverLink}
            onNodeHover={setHoverNode}
            
            // --- STYLING ---
            linkColor={() => "rgba(239, 68, 68, 0.6)"}
            linkWidth={link => (link === hoverLink || link.source === hoverNode || link.target === hoverNode) ? 5 : 1.2}
            linkCurvature={0.1}
            
            // BIGGER ARROWS
            linkDirectionalArrowLength={6}
            linkDirectionalArrowRelPos={0.5} 
            linkDirectionalArrowColor={() => "#ef4444"}
            
            // Unified Node Color
            // nodeLabel="label"
            nodeColor={() => "#f8fafc"}

            // DRAW LABELS LAST
            onRenderFramePost={(ctx, globalScale) => {
              graphData.links.forEach(link => {
                const isHovered = (link === hoverLink) || (link.source === hoverNode) || (link.target === hoverNode);
                if (!isHovered || !link.label) return;

                const start = link.source;
                const end = link.target;
                if (typeof start.x !== 'number' || typeof end.x !== 'number') return;

                const textX = start.x + (end.x - start.x) / 2;
                const textY = start.y + (end.y - start.y) / 2;

                // SCALED READABLE FONT
                const fontSize = 14 / globalScale; 
                ctx.font = `bold ${fontSize}px "Courier New", monospace`;
                const textWidth = ctx.measureText(link.label).width;
                const bckgDimensions = [textWidth + 6, fontSize + 4];

                ctx.save();
                ctx.translate(textX, textY);
                
                // Noir Style Pill Background
                ctx.fillStyle = 'rgba(2, 6, 23, 0.95)';
                ctx.strokeStyle = '#ef4444';
                ctx.lineWidth = 1 / globalScale;
                ctx.beginPath();
                ctx.roundRect(-bckgDimensions[0] / 2, -bckgDimensions[1] / 2, bckgDimensions[0], bckgDimensions[1], 2);
                ctx.fill();
                ctx.stroke();
                
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = '#ef4444'; 
                ctx.fillText(link.label, 0, 0);
                ctx.restore();
              });
            }}

            nodeCanvasObject={(node, ctx, globalScale) => {
              const label = node.label;
              const fontSize = 12 / globalScale;
              const radius = 5;
              const color = '#f8fafc';
              const isHovered = node === hoverNode;
              
              ctx.shadowColor = color;
              ctx.shadowBlur = isHovered ? 20 : 8;
              ctx.beginPath();
              ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
              ctx.fillStyle = color;
              ctx.fill();
              ctx.shadowBlur = 0; 

              if (globalScale > 1.2 || isHovered) {
                ctx.font = `bold ${fontSize}px "Courier New", monospace`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.strokeStyle = '#020617';
                ctx.lineWidth = 2 / globalScale;
                ctx.strokeText(label, node.x, node.y + radius + fontSize);
                ctx.fillStyle = color;
                ctx.fillText(label, node.x, node.y + radius + fontSize);
              }
            }}
          />
        </div>
      </div>
    </>
  );
}