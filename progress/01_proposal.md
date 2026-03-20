# Candidate Project Ideas

## 1) Healthcare - Patient Readmission + Anomaly Detection

Train a classifier to predict 30-day hospital readmission using either MIMIC-III or the Diabetes 130-US Hospitals dataset from UCI. Then layer on anomaly detection (Isolation Forest or an Autoencoder) to flag patients with unusual lab value profiles that do not fit typical readmission patterns. This combined approach identifies both high-risk patients and medically unusual cases.

## 2) Sports - Player Performance Regression + Clustering

Predict player salary or season statistics (points, WAR, etc.) using regression on NBA/MLB data from Basketball-Reference or Baseball Savant. Use clustering to discover player archetypes (for example, stretch-4s or two-way wings), then evaluate whether players are overpaid or underpaid relative to their archetype group. The FiveThirtyEight RAPTOR dataset is a strong option for this.

## 3) Airlines - Flight Delay Prediction + Anomaly Detection

Use the BTS On-Time Performance dataset (available at transtats.bts.gov) to predict delay severity with regression or classify delay causes. Add anomaly detection to identify unusual delay patterns, such as routes or time windows with statistically abnormal cancellation spikes that supervised models do not explain well. This provides a strong real-world framing around operational resilience.

## How AI influenced our idea selection

AI helped us refine and strengthen our project direction rather than choosing it for us. From the start, we knew we wanted to focus on airline data because of its practical relevance and our interest in the domain. The main gap was identifying a meaningful unsupervised learning component for the dataset. AI support (specifically Claude) helped surface anomaly detection as a strong fit, which gave us a clear way to combine supervised prediction with unsupervised pattern discovery.

AI also produced an initial implementation in `airline_test.py`, which gives us a useful starting point for development. Going forward, we plan to use AI in a guided, step-by-step way: we will define the goals and decisions, then use AI to help implement and iterate on code that aligns with our project plan.

## AI Excerpt / Exchange

> I found this dataset and think it'll work well to solve this problem. Give me a short overview of how I could use this in the project.  
> https://www.kaggle.com/datasets/sriharshaeedala/airline-delay/data  
>
> 9:49 PM
>
> Fetched: Flight Delay Data  
> Fetched: Flight Delay Data  
> Kaggle requires authentication to fetch dataset details, so I can't read the page directly. Let me search for it instead.
>
> Searched the web  
> Searched the web  
> Good - the schema is confirmed via a GitHub project that used this exact dataset. Here's a focused overview.
>
> What this dataset is  
> The dataset covers U.S. airport arrivals and delays by carrier, with columns including year, month, carrier, carrier_name, airport, airport_name, arr_flights, arr_del15, carrier_ct, weather_ct, nas_ct, security_ct, late_aircraft_ct, arr_cancelled, arr_diverted, arr_delay, and carrier_delay.  
> BTS Transtats
>
> The key thing to understand is that this is already aggregated - each row is not an individual flight but a carrier x airport x month summary. That's a meaningful structural difference from the BTS raw data discussed earlier, and it shapes how you'd use it for both parts of the project.
>
> How it fits the supervised learning component  
> Because each row is a monthly aggregate, your supervised target shifts from "will this flight be delayed?" to "what is the delay rate for this carrier at this airport this month?" That's a regression problem: predict arr_delay (total delay minutes) or derive a delay rate by dividing arr_del15 by arr_flights.

## Final Proposal

**Research question:** Can historical carrier-level delay statistics predict the likelihood of a high-delay month at a given airport, and can unsupervised anomaly detection identify carrier-airport combinations that are behaving outside their normal operational patterns?

**Supervised analysis:** The primary target is binary classification - whether a carrier-airport-month exceeds a 20% rate of significantly delayed arrivals (`arr_del15 / arr_flights > 0.20`), evaluated with AUROC. A secondary regression target predicting total delay minutes provides richer signal for the anomaly correlation analysis. XGBoost will be the primary model with logistic regression as a baseline, using the cause-breakdown columns (weather, NAS, carrier, late aircraft) as features alongside month and carrier/airport identity.

**Dataset:** The primary dataset is the Kaggle Airline Delay dataset ([kaggle.com/datasets/sriharshaeedala/airline-delay](https://www.kaggle.com/datasets/sriharshaeedala/airline-delay)), which provides pre-aggregated carrier x airport x month records with cause breakdowns already computed - meaning each row is directly usable as both a supervised training example and an anomaly detection input vector. The backup is the raw BTS On-Time Performance data at [transtats.bts.gov](https://transtats.bts.gov), which offers flight-level granularity if finer temporal resolution is needed.

**Feasibility:** The aggregated Kaggle dataset is small enough to train in seconds on a laptop with no GPU required. The full pipeline - EDA, supervised model, anomaly detection, joint analysis, and writeup - is scoped to 4-5 weeks. The monthly aggregation level intentionally constrains scope while still supporting meaningful analysis, including flagging events like the December 2022 Southwest collapse.

**Ethics and unsupervised method:** The dataset contains no PII and derives from U.S. government open data, so there are no privacy or licensing concerns. Anomaly findings will be framed analytically rather than as fault attribution to avoid misrepresenting correlational results as carrier performance judgments. For the unsupervised component, Isolation Forest will run on normalized cause-proportion and cancellation-rate features to flag anomalous carrier-airport-months. The key analytical contribution is the cross-tabulation of classifier output against anomaly scores - specifically demonstrating that supervised model errors increase as anomaly scores rise, establishing the detector as a practical reliability flag for the classifier.

