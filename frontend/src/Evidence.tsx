import type { Benchmark } from "./api";

export function Evidence({ benchmark }: { benchmark: Benchmark }) {
  const comparisons = Object.entries(benchmark.comparisons ?? {});
  if (!comparisons.length)
    return (
      <p className="note">
        Paired uncertainty estimates are unavailable for this run.
      </p>
    );
  return (
    <section className="evidence" aria-labelledby="uncertainty-title">
      <span className="eyebrow">LOOK BEYOND THE AVERAGE</span>
      <h2 id="uncertainty-title">How certain is the difference?</h2>
      <p>
        Paired nDCG@10 differences across the same queries. An interval crossing
        zero leaves the direction of the difference inconclusive.
      </p>
      <div className="evidence-grid">
        {comparisons.map(([name, comparison]) => {
          const [low, high] = comparison.ci95;
          const conclusion =
            low > 0
              ? "Positive interval"
              : high < 0
                ? "Negative interval"
                : "Inconclusive";
          return (
            <article key={name}>
              <h3>{name}</h3>
              <strong>{signed(comparison.mean_difference)}</strong>
              <span className="evidence-label">{conclusion}</span>
              <p>
                95% interval [{signed(low)}, {signed(high)}]
              </p>
              <small>
                {comparison.queries} queries ·{" "}
                {comparison.bootstrap_samples.toLocaleString()} resamples · seed{" "}
                {comparison.seed}
              </small>
            </article>
          );
        })}
      </div>
      <p className="note">
        Paired percentile bootstrap; exploratory comparisons without a
        multiple-testing correction. These intervals do not establish
        performance on other datasets. Latency measurements are a single warmed
        CPU pass, not a service guarantee.
      </p>
    </section>
  );
}
const signed = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
