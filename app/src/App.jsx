/**
 * @file App.jsx
 * @description Main application component for the Red String investigation board.
 * Renders a force-directed graph to visualize entity relationships extracted via LLM inference.
 * Includes entity resolution to merge semantic duplicates using cmpstr.
 */
import React, { useState, useRef, useEffect } from 'react';
// import { deduplicateTriplets } from './utils/deduplicate';
import ForceGraph2D from 'react-force-graph-2d';
import { Play, Activity } from 'lucide-react';
import { CmpStr } from 'cmpstr';
import { forceX, forceY } from 'd3-force';

// ---------------------
// --- CONFIGURATION ---
// ---------------------
const API_URL = "https://jets-daisy-thee-serve.trycloudflare.com/v1/completions";
// ---------------------
// ---------------------
// ---------------------

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

  const workerRef = useRef(null);

  const addLog = (msg) => setLogs(prev => [msg, ...prev]);

  useEffect(() => {
    // Vite specific syntax for loading a worker
    workerRef.current = new Worker(new URL('./utils/worker.js', import.meta.url), {
      type: 'module'
    });

    return () => {
      workerRef.current.terminate(); // Clean up on unmount
    };
  }, []);

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
          chargeForce.strength(-50); 
          chargeForce.distanceMax(110); 
        }
        fgRef.current.d3Force('center', null);

        fgRef.current.d3Force('x', forceX(0).strength(0.015));
        fgRef.current.d3Force('y', forceY(0).strength(0.015));
      }
    }, 100);
    return () => clearTimeout(timeout);
  }, []);

  const runDeduplication = (triplets, threshold = 0.75) => {
    return new Promise((resolve) => {
      const handleMessage = (e) => {
        if (e.data.status === 'complete') {
          workerRef.current.removeEventListener('message', handleMessage);
          resolve(e.data.result);
        }
      };

      workerRef.current.addEventListener('message', handleMessage);
      workerRef.current.postMessage({ triplets, threshold });
    });
  };


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
              .replace(/\b(inc|corp|llc|the|company|co|ltd)\b/g, "")
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

    const inputWords = normalizedInput.split(' ');
    if (inputWords.length > 5) return null;

    let bestMatch = null;
    let highestScore = 0;

    const cmp = CmpStr.create().setMetric('levenshtein').setFlags('i');
    const inputIsCapitalized = /^[A-Z]/.test(rawLabel.trim());

    // Generic exclusionary dictionary to prevent broad categorical merges
    const excludedGenerics = new Set(["us", "usa", "uk", "president", "state", "government", "department", "city", "county", "minister", "director", "secretary", "party", "union", "agency"]);

    for (const node of existingNodes) {
      const normalizedExisting = normalizeEntity(node.label);
      if (!normalizedExisting) continue;
      
      const existingWords = normalizedExisting.split(' ');
      if (existingWords.length > 5) continue;

      if (normalizedInput === normalizedExisting) return node;

      const existingIsCapitalized = /^[A-Z]/.test(node.label.trim());

      if (inputWords.length === 1 && existingWords.length > 1 && inputIsCapitalized) {
        const acronym = existingWords.map(w => w[0]).join('');
        if (normalizedInput === acronym) return node;
      } else if (existingWords.length === 1 && inputWords.length > 1 && existingIsCapitalized) {
        const acronym = inputWords.map(w => w[0]).join('');
        if (normalizedExisting === acronym) return node;
      }

      const isSubset = inputWords.every(w => existingWords.includes(w)) ||
                       existingWords.every(w => inputWords.includes(w));

      if (isSubset) {
        const wordDiff = Math.abs(inputWords.length - existingWords.length);
        if (wordDiff <= 1) {
          if (inputWords.length === 1 && !inputIsCapitalized && existingIsCapitalized) continue;
          if (existingWords.length === 1 && !existingIsCapitalized && inputIsCapitalized) continue;
          
          // Block subset merges if the isolated word is a generic noun
          if (inputWords.length === 1 && excludedGenerics.has(normalizedInput)) continue;
          if (existingWords.length === 1 && excludedGenerics.has(normalizedExisting)) continue;

          return node;
        }
      }

      if (Math.abs(normalizedInput.length - normalizedExisting.length) > 4) continue;

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

    if (highestScore > 0.90) return bestMatch;
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

        // Exclusionary filter
        if (t.head.split(' ').length > 5 || t.tail.split(' ').length > 5) return;

        let headNode = resolveEntity(t.head, newNodes);
        let tailNode = resolveEntity(t.tail, newNodes);

        // FIX: Contextual Spawning Logic
        // Default spawn near the center if it's a completely new isolated island
        let spawnX = (Math.random() - 0.5) * 50;
        let spawnY = (Math.random() - 0.5) * 50;

        // If one node already exists, spawn the new node right next to it
        if (headNode && headNode.x !== undefined && !tailNode) {
          spawnX = headNode.x + (Math.random() - 0.5) * 30;
          spawnY = headNode.y + (Math.random() - 0.5) * 30;
        } else if (tailNode && tailNode.x !== undefined && !headNode) {
          spawnX = tailNode.x + (Math.random() - 0.5) * 30;
          spawnY = tailNode.y + (Math.random() - 0.5) * 30;
        }

        // Process Head Entity
        if (!headNode) {
          headNode = { 
            id: t.head.toLowerCase().trim(), 
            label: t.head, 
            group: 1,
            x: spawnX, 
            y: spawnY 
          };
          newNodes.push(headNode);
        } else {
          if (t.head.toLowerCase().trim() !== headNode.label.toLowerCase().trim()) {
            console.log(`Resolved Entity [Head]: "${t.head}" merged into "${headNode.label}"`);
            addLog("Entity resolved!");
          }
          if (t.head.length > headNode.label.length) headNode.label = t.head; 
        }

        // Process Tail Entity
        if (!tailNode) {
          tailNode = { 
            id: t.tail.toLowerCase().trim(), 
            label: t.tail, 
            group: 2,
            x: spawnX, 
            y: spawnY 
          };
          newNodes.push(tailNode);
        } else {
          if (t.tail.toLowerCase().trim() !== tailNode.label.toLowerCase().trim()) {
            console.log(`Resolved Entity [Tail]: "${t.tail}" merged into "${tailNode.label}"`);
            addLog("Entity resolved!");
          }
          if (t.tail.length > tailNode.label.length) tailNode.label = t.tail;
        }

        const type = t.type.toLowerCase().trim();

        // Prevent duplicate links
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

  // const startInvestigation = async () => {
  //   if (!inputText) return;
  //   setIsProcessing(true);
  //   addLog("Starting Investigation...");

  //   const segmenter = new Intl.Segmenter('en', { granularity: 'sentence' });
  //   const segments = Array.from(segmenter.segment(inputText));
  //   const sentences = segments.map(s => s.segment.trim()).filter(s => s.length > 0);

  //   const windows = [];
  //   for (let i = 0; i < sentences.length; i++) {
  //     const current = sentences[i];
  //     const next = sentences[i + 1] || ""; 
  //     windows.push(`${current} ${next}`.trim());
  //   }

  //   addLog(`Text chunked into ${windows.length} segments.`);

  //   for (let i = 0; i < windows.length; i++) {
  //     const windowText = windows[i];
  //     addLog(`Scanning Window ${i+1}/${windows.length}...`);

  //     try {
  //       const payload = {
  //         prompt: `### Instruction:\nExtract all entity relationships from the following text and output them as a JSON list of triples.\n\n### Input:\n${windowText}\n\n### Response:\n`,
  //         max_tokens: 512,
  //         stop: ["###"]
  //       };

  //       const response = await fetch(API_URL, {
  //         method: "POST",
  //         headers: { 
  //           "Content-Type": "application/json",
  //         },
  //         body: JSON.stringify(payload)
  //       });

  //       const data = await response.json();
  //       const rawText = data.choices[0].message ? data.choices[0].message.content : data.choices[0].text;
  //       const triples = extractJSON(rawText);
        
  //       if (triples && triples.length > 0) {
  //         addLog(`Extracted ${triples.length} raw triples. Applying semantic deduplication...`);
  //         const cleanTriples = await runDeduplication(triples);
  //         addLog(`Kept ${cleanTriples.length} unique triples after deduplication.`);
  //         updateGraph(cleanTriples);
  //       }

  //     } catch (err) {
  //       addLog(`Error: ${err.message}`);
  //     }
  //   }

  //   addLog("Performing final global deduplication...");
    
  //   // 1. Collect all relationships currently in the graph.
  //   // After the graph renders, link.source and link.target become objects.
  //   const currentTriplets = graphData.links.map(l => ({
  //       head: l.source.label || l.source.id || l.source,
  //       type: l.type,
  //       tail: l.target.label || l.target.id || l.target
  //   }));

  //   if (currentTriplets.length > 0) {
  //       // 2. Run deduplication on the ENTIRE set using a more aggressive threshold (0.70)
  //       const finalCleanTriplets = await runDeduplication(currentTriplets, 0.70);
        
  //       addLog(`Global deduplication: merged ${currentTriplets.length} threads down to ${finalCleanTriplets.length}.`);

  //       // 3. Completely reset the graph and rebuild it with the cleaned master list.
  //       // Re-initializing prevents ghost nodes/links from windows that were merged.
  //       setGraphData({ nodes: [], links: [] });
        
  //       // Use a timeout to ensure React processes the reset before we rebuild.
  //       setTimeout(() => {
  //           updateGraph(finalCleanTriplets);
  //           setIsProcessing(false);
  //           addLog("Investigation Complete.");
  //       }, 200);
  //   } else {
  //       setIsProcessing(false);
  //       addLog("Investigation Complete.");
  //   }
  // };
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

    // Local array to store ALL raw triplets found across all windows
    let allTripletsAccumulator = [];

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
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        const data = await response.json();
        const rawText = data.choices[0].message ? data.choices[0].message.content : data.choices[0].text;
        const triples = extractJSON(rawText);
        
        if (triples && triples.length > 0) {
          // Add raw triples directly to the master list
          allTripletsAccumulator.push(...triples);
          
          // Show live progress on the graph (will be cleaned later)
          updateGraph(triples);
        }

      } catch (err) {
        addLog(`Error: ${err.message}`);
      }
    }

    // Perform global deduplication on all collected triplets
    if (allTripletsAccumulator.length > 0) {
        addLog(`Performing strict global deduplication on all ${allTripletsAccumulator.length} threads...`);
        
        // 0.90 merges highly semantically similar concepts
        const finalCleanTriplets = await runDeduplication(allTripletsAccumulator, 0.90);
        
        addLog(`Global deduplication complete: ${finalCleanTriplets.length} unique threads identified.`);

        // Reset and rebuild the graph
        setGraphData({ nodes: [], links: [] });
        
        setTimeout(() => {
            updateGraph(finalCleanTriplets);
            setIsProcessing(false);
            addLog("Investigation Complete.");
        }, 300);
    } else {
        setIsProcessing(false);
        addLog("Investigation Complete. No relationships found.");
    }
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