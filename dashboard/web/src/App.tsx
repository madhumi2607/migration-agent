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
    <div className="app">
      <header className="topbar">
        <a href="/" onClick={(e) => { e.preventDefault(); navigate("/"); }} className="brand">
          Migration Agent
        </a>
        <span className="muted">run viewer</span>
      </header>
      <main>{runMatch ? <RunPage runId={decodeURIComponent(runMatch[1])} /> : <RunsPage />}</main>
    </div>
  );
}
