import { useEffect, useState } from "react";
import RunPage from "./RunPage";
import RunsPage from "./RunsPage";

// Two routes, so a few lines of history handling instead of a router dependency:
//   /               -> runs list
//   /runs/<run-id>  -> one run
function currentPath() {
  return window.location.pathname;
}

export function navigate(path: string) {
  window.history.pushState(null, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
  window.scrollTo(0, 0);
}

export default function App() {
  const [path, setPath] = useState(currentPath);

  useEffect(() => {
    const onChange = () => setPath(currentPath());
    window.addEventListener("popstate", onChange);
    return () => window.removeEventListener("popstate", onChange);
  }, []);

  const runMatch = path.match(/^\/runs\/([^/]+)$/);
  return (
    <div className="shell">
      <aside className="sidebar">
        <a href="/" onClick={(e) => { e.preventDefault(); navigate("/"); }} className="brand">
          <span className="brand-mark" aria-hidden>⇡</span>
          <span>
            Migration Agent
            <span className="brand-sub">run viewer</span>
          </span>
        </a>
        <nav className="side-nav">
          <a href="/" onClick={(e) => { e.preventDefault(); navigate("/"); }} className={runMatch ? "" : "active"}>
            Runs
          </a>
        </nav>
        <div className="sidebar-foot muted small">Reads <span className="mono">runs/</span> · updates live</div>
      </aside>
      <main className="content">
        {runMatch ? <RunPage runId={decodeURIComponent(runMatch[1])} /> : <RunsPage />}
      </main>
    </div>
  );
}
