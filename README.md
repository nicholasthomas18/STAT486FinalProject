# STAT486FinalProject

## Final Proposal

**Research question:** Can historical carrier-level delay statistics predict the likelihood of a high-delay month at a given airport, and can unsupervised anomaly detection identify carrier-airport combinations that are behaving outside their normal operational patterns?

**Supervised analysis:** The primary target is binary classification - whether a carrier-airport-month exceeds a 20% rate of significantly delayed arrivals (`arr_del15 / arr_flights > 0.20`), evaluated with AUROC. A secondary regression target predicting total delay minutes provides richer signal for the anomaly correlation analysis. XGBoost will be the primary model with logistic regression as a baseline, using the cause-breakdown columns (weather, NAS, carrier, late aircraft) as features alongside month and carrier/airport identity.

**Dataset:** The primary dataset is the Kaggle Airline Delay dataset ([kaggle.com/datasets/sriharshaeedala/airline-delay](https://www.kaggle.com/datasets/sriharshaeedala/airline-delay)), which provides pre-aggregated carrier x airport x month records with cause breakdowns already computed - meaning each row is directly usable as both a supervised training example and an anomaly detection input vector. The backup is the raw BTS On-Time Performance data at [transtats.bts.gov](https://transtats.bts.gov), which offers flight-level granularity if finer temporal resolution is needed.

**Feasibility:** The aggregated Kaggle dataset is small enough to train in seconds on a laptop with no GPU required. The full pipeline - EDA, supervised model, anomaly detection, joint analysis, and writeup - is scoped to 4-5 weeks. The monthly aggregation level intentionally constrains scope while still supporting meaningful analysis, including flagging events like the December 2022 Southwest collapse.

**Ethics and unsupervised method:** The dataset contains no PII and derives from U.S. government open data, so there are no privacy or licensing concerns. Anomaly findings will be framed analytically rather than as fault attribution to avoid misrepresenting correlational results as carrier performance judgments. For the unsupervised component, Isolation Forest will run on normalized cause-proportion and cancellation-rate features to flag anomalous carrier-airport-months. The key analytical contribution is the cross-tabulation of classifier output against anomaly scores - specifically demonstrating that supervised model errors increase as anomaly scores rise, establishing the detector as a practical reliability flag for the classifier.
