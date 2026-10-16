import { useEffect, useState } from "react";
import { api } from "./api";

interface Study {
  anchors?: number;
  events?: number;
  seed_papers?: number;
  summary: Record<string, Record<string, number>>;
  comparison: {
    mean_difference: number;
    ci95: [number, number];
    groups: number;
  };
}
interface Studies {
  available: boolean;
  local?: Study;
  external?: Study;
}

export function RecommendationEvidence() {
  const [studies, setStudies] = useState<Studies | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    api<Studies>("/recommendation-benchmark", { signal: controller.signal })
      .then(setStudies)
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      });
    return () => controller.abort();
  }, []);
  return (
    <section className="evidence" aria-labelledby="related-evaluation-title">
      <span className="eyebrow">A SEPARATE RECOMMENDATION TEST</span>
      <h2 id="related-evaluation-title">How accurate are related papers?</h2>
      <p>
        These evaluations compare paper-to-paper recommendations. Their labels
        and candidate pools differ from the search benchmark above.
      </p>
      {error && <p role="alert">{error}</p>}
      {!studies && !error && (
        <p role="status">Loading recommendation evaluation…</p>
      )}
      {studies && !studies.available && (
        <p>
          No matching recommendation evaluation is available for this index.
        </p>
      )}
      {studies?.local && (
        <StudyTable
          study={studies.local}
          title="Local co-evidence proxy"
          metrics={["NDCG@5", "Recall@5", "Hit@5", "MRR@100"]}
          context={`${studies.local.anchors} test papers, searched against the full 5,183-paper corpus. Positives share a human-annotated evidence claim; they are not direct recommendation ratings.`}
        />
      )}
      {studies?.external && (
        <StudyTable
          study={studies.external}
          title="External logged-click test"
          metrics={["Adj-NDCG@10", "Adj-P@1", "Adj-MRR"]}
          context={`${studies.external.events} SciDocs test click events from ${studies.external.seed_papers} seed papers. Each event reranks only ten previously displayed candidates. Adjusted metrics use inverse display-position propensity weights.`}
        />
      )}
      {studies?.available && (
        <p className="note">
          Higher scores are better. The two tests are not directly comparable.
          Unknown relevance is not proof of irrelevance, and clicks are not
          complete user ratings. Settings were selected on paper-disjoint
          training proxy labels, then frozen. Scores are not relevance
          probabilities.{" "}
          <a
            href="https://github.com/singusathwik/ScholarMind-AI/blob/main/docs/RELATED_PAPERS_EVALUATION.md"
            target="_blank"
            rel="noreferrer"
          >
            Read the evaluation protocol and limitations.
          </a>
        </p>
      )}
    </section>
  );
}

function StudyTable({
  study,
  title,
  metrics,
  context,
}: {
  study: Study;
  title: string;
  metrics: string[];
  context: string;
}) {
  const [low, high] = study.comparison.ci95;
  return (
    <article>
      <h3>{title}</h3>
      <p>{context}</p>
      <div className="table-scroll">
        <table>
          <caption>{title}: original versus recommended blend</caption>
          <thead>
            <tr>
              <th>Method</th>
              {metrics.map((m) => (
                <th key={m}>{m}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(study.summary).map(([name, row]) => (
              <tr key={name}>
                <th>
                  {name === "dense" ? "Original semantic" : "Recommended blend"}
                </th>
                {metrics.map((m) => (
                  <td key={m}>{row[m].toFixed(4)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">
        Paired nDCG difference: {study.comparison.mean_difference.toFixed(4)};
        95% cluster-bootstrap interval [{low.toFixed(4)}, {high.toFixed(4)}],{" "}
        {study.comparison.groups} groups.{" "}
        {low > 0
          ? "Positive interval on this test."
          : high < 0
            ? "Negative interval on this test."
            : "Difference remains inconclusive."}{" "}
        Groups preserve linked-paper or repeated-click dependence.
      </p>
    </article>
  );
}
