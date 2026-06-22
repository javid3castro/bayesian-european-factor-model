# Development Log

This document records the iterative development of the model, including the hypotheses tested at each stage, the results obtained, and the reasoning behind decisions to retain, modify, or discard each approach. It is included to provide a transparent account of the research process rather than presenting only the final specification.

---

## Version 1: Base specification

The initial specification used five factors (earnings yield, return on equity, six-month momentum, realised volatility, one-month reversal) computed across the full 60-stock universe without sector adjustment, estimated via the Bayesian logistic regression and Metropolis-Hastings procedure described in the main methodology, and validated using expanding-window walk-forward cross-validation.

This specification produced a cumulative outperformance of 26.8 percentage points against the STOXX 600 over the evaluation period, with a quarterly hit rate of 64.0 percent. The base rate of the binary outcome variable was approximately 51.5 percent, confirming that the classification problem was reasonably balanced and not trivially solved by a constant prediction. Exploratory analysis indicated that all five factors exhibited the expected directional relationship with the outcome variable at the univariate level.

---

## Version 2: Correction of return-on-equity data quality

During exploratory analysis it was observed that the histogram density of return on equity was substantially lower than that of the other four factors (3.9 percent against a range of 13 to 37 percent for the remaining factors), indicating a high proportion of missing or unreliable values. Manual inspection identified specific cases of clearly erroneous figures obtained from the standard Yahoo Finance API field, most notably an apparent return on equity of 623 percent for one industrial issuer, attributable to near-zero or negative book equity following a period of asset impairment.

A more robust extraction procedure was implemented, computing return on equity directly from reported net income and shareholders' equity in the underlying financial statements rather than relying on the pre-computed API field. For five issuers where this calculation remained unreliable or unavailable, return on equity was set manually using verified figures from company reporting, and all values were subsequently capped to the range minus 100 percent to plus 150 percent to limit the influence of any remaining outliers.

This correction improved cumulative outperformance to 30.5 percentage points, an increase of 3.7 percentage points over the base specification, and increased return-on-equity histogram density by approximately 250 percent. The finding illustrates that, for a universe of this size, the integrity of a single factor's underlying data can materially affect the estimated coefficient and the resulting portfolio.

---

## Version 3: Sector-neutral factor construction

A further hypothesis was tested: that ranking stocks against the full cross-section conflates genuine relative mispricing with persistent sector-level differences in valuation multiples. Telecommunications issuers, for example, typically trade at structurally higher earnings yields than consumer luxury issuers as a function of sector characteristics rather than mispricing, and a model that does not account for this will tend to overweight entire sectors rather than identifying genuine cross-sectional anomalies.

Earnings yield and return on equity were re-computed as within-sector z-scores across eleven sector groupings, each requiring a minimum of three constituent stocks. A representative example of the effect: one telecommunications issuer exhibited a raw earnings yield of 11.3 percent, which in isolation appeared attractive, but a within-sector z-score of minus 1.17 indicated that the issuer was in fact expensive relative to its telecommunications peer group.

This specification produced the strongest result obtained in the project: cumulative outperformance of 34.0 percentage points and a quarterly hit rate of 62.9 percent. The momentum factor's univariate predictive strength, measured by the ratio of positive to negative decile outcomes, increased from 1.87 to 3.09. This specification is retained as the production model and is the version documented in the main methodology section of this repository.

---

## Version 4: Universe expansion (discarded)

It was hypothesised that expanding the investable universe from 60 to approximately 150 stocks would improve the reliability of the sector-neutral z-scores, since several sector groupings in the 60-stock universe contained as few as two or three constituents.

This specification produced a materially worse result: cumulative outperformance fell to 22.7 percentage points and the hit rate fell to 55.7 percent. Diagnostic analysis indicated that the additional approximately 90 stocks, drawn predominantly from smaller-capitalisation issuers across the four exchanges, carried substantially less reliable fundamental data through the same free data source; the return-on-equity factor's univariate predictive strength fell from a ratio of 3.77 to 1.63 as a direct consequence.

The specification was reverted to the 60-stock universe. This result is recorded as a deliberate negative finding: it would have been straightforward to adopt the larger universe without identifying the underlying cause of the performance deterioration, and the experiment demonstrates that, in the presence of a free data source with uneven coverage, universe size is constrained by data quality rather than by model capacity.

---

## Version 5: Earnings revision factor (not adopted)

A sixth factor was tested, constructed from the ratio of consensus forward to trailing earnings per share, intended to capture the earnings revision anomaly documented by Chan, Jegadeesh and Lakonishok (1996). In univariate testing this factor produced the strongest predictive signal of any factor considered in the project, with a positive-to-negative decile ratio of 5.28, exceeding both return on equity and momentum.

Despite this, incorporating the factor into the production specification reduced cumulative outperformance to 22.7 percentage points. The cause was identified as a data limitation rather than a deficiency in the factor itself: the data source provides only the current consensus estimate rather than a point-in-time historical series, meaning that the same present-day analyst sentiment was, in effect, being applied across the entire 2016-2024 backtest period rather than the sentiment that would have been observable at each historical point in time. This introduces a data-staleness bias distinct from conventional look-ahead bias, since no future return information is used, but the factor input itself is not historically accurate.

The factor is not included in the production specification but is recorded as the highest-priority candidate for future inclusion should a point-in-time fundamental data source become available.

---

## Version 6: Bayesian Kelly position sizing (discarded)

Equal weighting across the top quartile of ranked stocks was replaced with position sizing proportional to posterior conviction, following the Bayesian Kelly criterion of Browne (1996): position size proportional to (posterior mean probability minus one half) divided by the posterior standard deviation, scaled by a fractional Kelly parameter of one half.

This specification underperformed equal weighting substantially, reducing cumulative outperformance to 18.4 percentage points. Diagnostic analysis identified the cause: a well-converged Markov chain with 12,000 post-burn-in draws produces posterior standard deviations on the order of 0.04, an order of magnitude smaller than the values typically assumed in textbook applications of the Kelly criterion. Dividing by a denominator of this magnitude produced extreme raw position sizes that were then truncated by the position cap in an economically arbitrary manner, concentrating capital on stocks with low posterior uncertainty rather than high expected edge.

---

## Version 7: Volatility-adjusted position sizing

The position-sizing denominator was replaced with realised stock-level volatility in place of posterior standard deviation, removing the scaling problem identified in Version 6 while retaining the principle of conviction-weighted sizing: position size proportional to (posterior mean probability minus one half) divided by trailing realised volatility.

This specification improved cumulative outperformance to 27.0 percentage points relative to Version 6, though it remained below the 34.0 percentage points achieved by equal weighting in Version 3. The quarterly hit rate, however, increased to 80.9 percent, the highest of any specification tested, reflecting a larger number of more diversified, smaller positions rather than the more concentrated positions implied by equal weighting within a fixed top quartile.

This result is interpreted as a genuine risk-return trade-off rather than a deficiency in either approach: equal weighting concentrates capital and produces higher absolute cumulative alpha at a lower hit rate, while volatility-adjusted sizing diversifies capital and produces a substantially more consistent quarter-by-quarter result at a lower absolute return. The choice between the two would reasonably depend on an allocator's specific risk preferences. The production model retains equal weighting (Version 3); the volatility-adjusted specification (Version 7) is documented as a viable lower-volatility alternative.

## Version 8: Data integrity investigation and point-in-time remediation

This entry documents the most significant methodological finding of the project, arising not from a planned experiment but from an unexpected reproducibility failure.

The model was re-executed, with no code changes, several weeks after the run that produced the Version 3 results reported as the production specification. The posterior coefficient for earnings yield, which had been positive (approximately 0.41) and statistically significant in the original run, was found to be negative (approximately minus 0.23) and statistically significant in the second run. Investigation traced this to the fundamental dataset: earnings yield and return on equity were downloaded once per execution and applied identically to all 35 quarters of the backtest spanning 2016 to 2024, meaning the values used to evaluate, for instance, the first quarter of 2017 reflected whatever the data happened to be on the day the script was run, often years later. The coefficient sign was consequently sensitive to execution timing rather than reflecting a stable empirical relationship.

A point-in-time data pipeline was constructed to address this directly. For each stock, the full available history of quarterly and annual financial statements was retrieved, and each reported figure was assigned an availability date equal to its period-end date plus a 90-day reporting lag. Quarterly factor construction was modified to perform an as-of lookup against this history at each point in the backtest, using only data that would genuinely have been public knowledge at that time. Sector-neutral standardisation, previously computed once across the full sample, was moved inside the per-quarter loop and computed independently for each quarter using only the stocks with valid point-in-time data available in that quarter.

During implementation a second error was found and corrected: the quarterly standardisation function initially filled sector-quarter groups with no valid observations with a value of zero, rather than leaving them undefined. This silently fabricated factor values for periods with no underlying data and produced an apparently complete dataset reaching back to 2016, obscuring the limitation that the fix was intended to expose. Correcting this revealed that, under genuine point-in-time discipline, only 17.3 percent of stock-quarter observations carry a valid earnings yield and 19.8 percent a valid return on equity, with usable data concentrated almost entirely between the third quarter of 2022 and the third quarter of 2024. Of the 35 quarters in the full price history, only 9 have usable point-in-time fundamental data, a direct consequence of the limited historical depth of free financial statement data available through Yahoo Finance rather than any property of the model itself.

Nine quarters does not constitute a sufficient sample for a walk-forward validation procedure to yield statistically meaningful inference, and no performance metrics are reported for the point-in-time specification for this reason. The practical implication is that the headline performance figures reported for Version 3 throughout this repository, including the 34 percentage point cumulative outperformance result, should be understood as a demonstration of the modelling and validation methodology under a simplifying data assumption, not as a validated estimate of achievable performance, and the reproducibility failure that motivated this investigation indicates that the specific magnitude and sign of those results should not be relied upon. Acquisition of institutional point-in-time fundamental data, for example through Bloomberg, Refinitiv, or a lower-cost alternative such as Simfin+, is accordingly treated as a precondition for any future attempt to validate this model's performance with statistical confidence, and is the single highest-priority item for further work. The point-in-time pipeline itself required no further modification once a sufficiently deep data source is substituted.

---

## Summary of open research questions

The following extensions were identified during development but have not yet been implemented or tested:

Acquisition of institutional-grade point-in-time fundamental data, identified by the investigation in Version 8 as a precondition for statistically meaningful validation of this model, rather than a discretionary enhancement.

Monthly rather than quarterly rebalancing, which may capture a larger share of the momentum signal given the factor's documented six- to twelve-month horizon.

Incorporation of the earnings revision factor (Version 5) contingent on access to a point-in-time fundamental data source.

Explicit transaction cost modelling, since all results reported in this repository are gross of trading costs.

Out-of-sample evaluation on data from 2025 onward, which has not been used in any prior specification or parameter selection decision and would therefore constitute a genuinely held-out test.
