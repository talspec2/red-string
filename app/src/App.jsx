/**
 * @file App.jsx
 * @description Main application component for the Red String investigation board.
 * Renders a force-directed graph to visualize entity relationships extracted via LLM inference.
 * Includes entity resolution to merge semantic duplicates using cmpstr.
 */
import React, { useState, useRef, useEffect } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Play, Activity } from 'lucide-react';
import { CmpStr } from 'cmpstr';

// --- CONFIGURATION ---
const API_URL = "https://titles-soundtrack-models-respective.trycloudflare.com/v1/completions";

export default function RedStringApp() {
  const [inputText, setInputText] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [logs, setLogs] = useState([]);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  
  const [hoverLink, setHoverLink] = useState(null);
  const [hoverNode, setHoverNode] = useState(null);
  
  const graphWrapperRef = useRef(null);
  const fgRef = useRef(); 
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  const addLog = (msg) => setLogs(prev => [msg, ...prev]);

  useEffect(() => {
    const resizeObserver = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });
    if (graphWrapperRef.current) resizeObserver.observe(graphWrapperRef.current);
    return () => resizeObserver.disconnect();
  }, []);

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

  /**
   * Normalizes an entity string for clean comparison.
   * Removes punctuation, extra spaces, and common corporate suffixes.
   */
  const normalizeEntity = (str) => {
    if (!str) return "";
    return str.toLowerCase()
              .replace(/[.,/#!$%^&*;:{}=\-_`~()]/g, "")
              .replace(/\b(inc|corp|llc|the|company|co)\b/g, "")
              .replace(/\s{2,}/g, " ")
              .trim();
  };

  /**
   * Resolves a raw entity label against existing nodes to prevent duplication.
   * Utilizes cmpstr for fuzzy matching to calculate a normalized similarity score.
   */
  const resolveEntity = (rawLabel, existingNodes) => {
    const normalizedInput = normalizeEntity(rawLabel);
    if (!normalizedInput || normalizedInput.length < 2) return null;

    let bestMatch = null;
    let highestScore = 0;

    // Initialize cmpstr with the Levenshtein metric and case-insensitive flags
    const cmp = CmpStr.create().setMetric('levenshtein').setFlags('i');

    for (const node of existingNodes) {
      const normalizedExisting = normalizeEntity(node.label);
      if (!normalizedExisting) continue;
      
      // 1. Exact normalized match
      if (normalizedInput === normalizedExisting) {
        return node;
      }

      // 2. Substring match
      if (normalizedInput.length > 3 && normalizedExisting.length > 3) {
        if (normalizedExisting.includes(normalizedInput) || normalizedInput.includes(normalizedExisting)) {
          return node;
        }
      }

      // 3. Fuzzy match
      try {
        const result = cmp.test([normalizedInput], normalizedExisting);
        if (result && result.match > highestScore) {
          highestScore = result.match;
          bestMatch = node;
        }
      } catch (e) {
        continue;
      }
    }

    // Threshold for accepting a fuzzy match
    if (highestScore > 0.85) { 
      return bestMatch;
    }

    return null;
  };

  const updateGraph = (triples) => {
    if (!triples || triples.length === 0) return;

    setGraphData(prev => {
      const newNodes = [...prev.nodes];
      const newLinks = [...prev.links];
      let addedCount = 0;

      triples.forEach(t => {
        if (!t.head || !t.tail || !t.type) return;

        let headNode = resolveEntity(t.head, newNodes);
        if (!headNode) {
          headNode = { id: t.head.toLowerCase().trim(), label: t.head, group: 1 };
          newNodes.push(headNode);
        } else if (t.head.length > headNode.label.length) {
          headNode.label = t.head; 
        }

        let tailNode = resolveEntity(t.tail, newNodes);
        if (!tailNode) {
          tailNode = { id: t.tail.toLowerCase().trim(), label: t.tail, group: 2 };
          newNodes.push(tailNode);
        } else if (t.tail.length > tailNode.label.length) {
          tailNode.label = t.tail;
        }

        const type = t.type.toLowerCase().trim();

        const exists = newLinks.some(l => 
          (l.source.id === headNode.id || l.source === headNode.id) && 
          (l.target.id === tailNode.id || l.target === tailNode.id) &&
          l.type === type
        );
        
        if (!exists) {
          newLinks.push({ source: headNode.id, target: tailNode.id, type: type, label: type });
          addedCount++;
        }
      });

      if (addedCount > 0) addLog(`Added ${addedCount} threads.`);
      return { nodes: newNodes, links: newLinks };
    });
  };

  const startInvestigation = async () => {
    if (!inputText) return;
    setIsProcessing(true);
    addLog("Starting Investigation...");

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
      addLog(`Scanning Window ${i+1}/${windows.length}...`);

      try {
        const payload = {
          prompt: `### Instruction:\nExtract all entity relationships from the following text and output them as a JSON list of triples.\n\n### Input:\n${windowText}\n\n### Response:\n`,
          max_tokens: 512,
          stop: ["###"]
        };

        const response = await fetch(API_URL, {
          method: "POST",
          headers: { 
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload)
        });

        const data = await response.json();
        const rawText = data.choices[0].message ? data.choices[0].message.content : data.choices[0].text;
        const triples = extractJSON(rawText);
        
        if (triples && triples.length > 0) updateGraph(triples);

      } catch (err) {
        addLog(`Error: ${err.message}`);
      }
    }
    setIsProcessing(false);
    addLog("Investigation Complete.");
  };

  return (
    <>
      <style>
        {`
          body, html, #root { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #0f172a; }
        `}
      </style>

      <div style={{ display: 'flex', width: '100vw', height: '100vh', background: '#0f172a', color: '#f8fafc', fontFamily: 'sans-serif' }}>
        
        <div style={{ width: '400px', borderRight: '1px solid #334155', display: 'flex', flexDirection: 'column', padding: '20px', backgroundColor: '#1e293b' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
            <Activity color="#ef4444" />
            <h1 style={{ fontSize: '1.2rem', margin: 0, fontWeight: 'bold' }}>Red String Investigator</h1>
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
            
            d3VelocityDecay={0.5} 
            cooldownTicks={150}   
            
            onLinkHover={setHoverLink}
            onNodeHover={setHoverNode}
            
            linkColor={() => "rgba(239, 68, 68, 0.6)"}
            linkWidth={link => (link === hoverLink || link.source === hoverNode || link.target === hoverNode) ? 5 : 1.2}
            linkCurvature={0.1}
            linkDirectionalArrowLength={6}
            linkDirectionalArrowRelPos={0.5} 
            linkDirectionalArrowColor={() => "#ef4444"}
            
            nodeColor={() => "#f8fafc"}

            onRenderFramePost={(ctx, globalScale) => {
              graphData.links.forEach(link => {
                const isHovered = (link === hoverLink) || (link.source === hoverNode) || (link.target === hoverNode);
                if (!isHovered || !link.label) return;

                const start = link.source;
                const end = link.target;
                if (typeof start.x !== 'number' || typeof end.x !== 'number') return;

                const textX = start.x + (end.x - start.x) / 2;
                const textY = start.y + (end.y - start.y) / 2;

                const fontSize = 14 / globalScale; 
                ctx.font = `bold ${fontSize}px "Courier New", monospace`;
                const textWidth = ctx.measureText(link.label).width;
                const bckgDimensions = [textWidth + 6, fontSize + 4];

                ctx.save();
                ctx.translate(textX, textY);
                
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