import { useEffect, useState, type CSSProperties } from "react";
import { Activity, BarChart3, Check, Download, FileImage, History, Moon, PanelLeft, Play, Settings, SlidersHorizontal, UploadCloud, X } from "lucide-react";
import { assetUrl, getHistory, health, process, upload } from "./services/api";
import type { HistoryItem, Method, ProcessResult } from "./types";

const methodLabels: Record<Method, string> = { rgb: "RGB channel", chrominance: "Chrominance", perceptual: "Perceptually constrained" };

function Loading({ status }: { status: string }) {
  return <div className="loading"><div className="scan-grid"><span /><span /><span /><span /><span /><span /><span /><span /><span /></div><div className="wordmark">VISION</div><p>Initializing image processing engine...</p><small>{status}</small><div className="loader-line"><i /></div></div>;
}

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return <div className="metric"><div><span>{label}</span><small>{note}</small></div><strong>{value}</strong></div>;
}

export default function App() {
  const [ready, setReady] = useState(false); const [status, setStatus] = useState("Connecting to backend");
  const [online, setOnline] = useState(false); const [file, setFile] = useState<File>(); const [preview, setPreview] = useState("");
  const [imageId, setImageId] = useState(""); const [result, setResult] = useState<ProcessResult>(); const [busy, setBusy] = useState(false);
  const [error, setError] = useState(""); const [method, setMethod] = useState<Method>("perceptual"); const [budget, setBudget] = useState(4);
  const [iterations, setIterations] = useState(1); const [seed, setSeed] = useState(42); const [tab, setTab] = useState("Workspace");
  const [dark, setDark] = useState(false); const [split, setSplit] = useState(50);
  const [runVersion, setRunVersion] = useState(0); const [viewMode, setViewMode] = useState<"modified" | "difference">("modified");
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);

  useEffect(() => { health().then(ok => { setOnline(ok); setStatus(ok ? "Engine ready" : "Backend unavailable — start FastAPI to continue"); setTimeout(() => setReady(true), 250); }); }, []);
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);
  useEffect(() => {
    if (tab === "Image History" && online) {
      getHistory().then(setHistoryItems).catch(() => setHistoryItems([]));
    }
  }, [tab, online, runVersion]);

  const metrics = result?.metrics;
  async function choose(next: File) {
    setError(""); setFile(next); setPreview(URL.createObjectURL(next)); setResult(undefined);
    try { const uploaded = await upload(next); setImageId(uploaded.image_id); } catch (e) { setError(e instanceof Error ? e.message : "Upload failed"); }
  }
  async function run() {
    if (!imageId) return setError("Upload an image before processing."); setBusy(true); setError("");
    try {
      const next = await process(imageId, file?.name ?? "image", method, budget, iterations, seed);
      setResult(next);
      setRunVersion(Date.now());
    } catch (e) { setError(e instanceof Error ? e.message : "Processing failed"); } finally { setBusy(false); }
  }
  if (!ready) return <Loading status={status} />;
  return <div className={dark ? "app dark" : "app"}>
    <aside className="sidebar"><div className="brand"><div className="brand-mark">V</div><span>VISION</span></div><div className="workspace-label">PROJECT / DEFAULT</div><nav>{[["Workspace", PanelLeft], ["Image History", History], ["Experiments", Activity], ["Analytics", BarChart3], ["Settings", Settings]].map(([label, Icon]) => <button className={tab === label ? "nav active" : "nav"} onClick={() => setTab(label as string)} key={label as string}><Icon size={16} />{label as string}</button>)}</nav><div className="sidebar-foot"><span className={online ? "status-dot" : "status-dot off"} />{online ? "Engine connected" : "Engine offline"}<button className="icon-btn" onClick={() => setDark(!dark)} title="Toggle theme"><Moon size={15} /></button></div></aside>
    <main><header><div><span className="eyebrow">WORKSPACE / {tab.toUpperCase()}</span><h1>{tab === "Workspace" ? "Perceptual image lab" : tab}</h1></div><div className="header-actions"><span className={online ? "connection" : "connection offline"}><span />{online ? "Backend online" : "Backend offline"}</span><button className="icon-btn" onClick={() => setTab("Settings")} title="Settings"><SlidersHorizontal size={16} /></button></div></header>
      {tab === "Image History" ? (
        historyItems.length > 0 ? <div className="history-panel"><table className="history-table"><thead><tr><th>File</th><th>Method</th><th>Budget</th><th>PSNR</th><th>SSIM</th><th>Delta E</th><th>Size</th><th>Duration</th></tr></thead><tbody>{historyItems.map(item => <tr key={item.id}><td>{item.filename}</td><td>{item.method}</td><td>{item.budget}</td><td>{item.metrics.psnr.toFixed(2)} dB</td><td>{item.metrics.ssim.toFixed(4)}</td><td>{item.metrics.delta_e_2000.toFixed(3)}</td><td>{item.width}×{item.height}</td><td>{item.duration_ms.toFixed(0)} ms</td></tr>)}</tbody></table></div>
          : <div className="empty-panel"><div className="empty-icon"><History size={22} /></div><h2>Image History</h2><p>Run an experiment from Workspace to populate this view with measured results.</p></div>
      ) : tab !== "Workspace" ? <div className="empty-panel"><div className="empty-icon"><FileImage size={22} /></div><h2>{tab}</h2><p>{tab === "Settings" ? "Application defaults and connection settings are managed through the local configuration." : "Run an experiment from Workspace to populate this view with measured results."}</p></div> :
      <div className="workspace"><section className="control-panel"><div className="section-head"><div><span className="section-index">01</span><h2>Input image</h2></div><span className="muted">JPG · PNG · WEBP / 15 MB</span></div><label className="dropzone" onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); const dropped = e.dataTransfer.files[0]; if (dropped) void choose(dropped); }}><input type="file" accept=".jpg,.jpeg,.png,.webp" onChange={e => e.target.files?.[0] && void choose(e.target.files[0])} /><UploadCloud size={22} /><strong>{file ? file.name : "Drop an image here"}</strong><span>{file ? `${(file.size / 1024).toFixed(1)} KB · ${file.type}` : "or browse from your computer"}</span><span className="browse">Browse files</span></label>
        <div className="section-head controls-head"><div><span className="section-index">02</span><h2>Perturbation parameters</h2></div></div><div className="form-grid"><label>Method<select value={method} onChange={e => setMethod(e.target.value as Method)}>{Object.entries(methodLabels).map(([v, l]) => <option value={v} key={v}>{l}</option>)}</select></label><label>Budget <output>{budget.toFixed(1)}</output><input type="range" min="0" max="16" step=".5" value={budget} onChange={e => setBudget(Number(e.target.value))} /><small>Maximum perceptual change</small></label><label>Iterations<input type="number" min="1" max="20" value={iterations} onChange={e => setIterations(Math.min(20, Math.max(1, Math.trunc(Number(e.target.value)) || 1)))} /></label><label>Random seed<input type="number" min="0" value={seed} onChange={e => setSeed(Math.max(0, Math.trunc(Number(e.target.value)) || 0))} /></label></div><button className="run-btn" disabled={!online || !imageId || busy} onClick={() => void run()}>{busy ? <><span className="spinner" /> Processing image</> : <><Play size={15} /> Run perturbation</>}</button>{error && <div className="error"><X size={15} />{error}</div>}</section>
        <section className="results"><div className="section-head"><div><span className="section-index">03</span><h2>Comparison</h2></div>{result && <div className="compare-actions"><button type="button" className={viewMode === "modified" ? "mode-btn active" : "mode-btn"} onClick={() => setViewMode("modified")}>Modified</button><button type="button" className={viewMode === "difference" ? "mode-btn active" : "mode-btn"} onClick={() => setViewMode("difference")}>Difference</button><a className="download" href={assetUrl(viewMode === "modified" ? result.modified_url : result.difference_url, runVersion)} download><Download size={14} /> Export PNG</a></div>}</div>{result ? <><div className="comparison" style={{ "--split": `${split}%` } as CSSProperties}><div className="compare-base" style={{ backgroundImage: `url(${assetUrl(result.original_url, runVersion)})` }} /><div className="compare-modified" style={{ backgroundImage: `url(${assetUrl(viewMode === "modified" ? result.modified_url : result.difference_url, runVersion)})`, clipPath: `inset(0 ${100 - split}% 0 0)` }} /><input aria-label="Comparison slider" type="range" min="0" max="100" value={split} onChange={e => setSplit(Number(e.target.value))} /><div className="compare-label left">ORIGINAL</div><div className="compare-label right">{viewMode === "modified" ? "MODIFIED" : "DIFFERENCE"}</div></div><div className="image-meta"><span>{result.width} × {result.height}px</span><span>{(result.original_size / 1024).toFixed(1)} KB → {(result.modified_size / 1024).toFixed(1)} KB</span><span>{result.processing_time_ms.toFixed(0)} ms</span></div><div className="section-head metrics-head"><h2>Measured similarity</h2><span className="verified"><Check size={13} /> Computed locally</span></div><div className="metrics"><Metric label="PSNR" value={`${metrics?.psnr.toFixed(2)} dB`} note="Signal fidelity" /><Metric label="SSIM" value={metrics?.ssim.toFixed(4) ?? "—"} note="Structural similarity" /><Metric label="Delta E 2000" value={metrics?.delta_e_2000.toFixed(3) ?? "—"} note="Mean color difference" /><Metric label="RGB difference" value={metrics?.mean_absolute_rgb_difference.toFixed(2) ?? "—"} note="Mean absolute value" /><Metric label="Changed pixels" value={`${metrics?.changed_pixel_percentage.toFixed(1)}%`} note="Any channel changed" /><Metric label="Max difference" value={`${metrics?.maximum_pixel_difference}`} note="Per-channel maximum" /></div></> : <div className="result-placeholder"><FileImage size={28} /><p>Processed output will appear here</p><small>Upload an image and run a perturbation to see the before / after comparison.</small></div>}</section></div>}</main>
  </div>;
}
