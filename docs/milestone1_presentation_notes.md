# Austin Traffic Speed Prediction
## Milestone I — Presentation Document

---

## SLIDE 1 — Project Goal, Audience, Purpose and Context

**Project Goal**

The goal of this project is to build a machine learning system that predicts vehicle speed at traffic sensor locations across Austin, Texas for up to 60 minutes into the future. The system takes 60 minutes of historical traffic observations as input and outputs speed predictions at 15-minute, 30-minute, and 60-minute horizons.

**Audience**

- City of Austin transportation planners who need to anticipate congestion before it worsens
- Navigation and routing applications that benefit from short-term speed forecasts
- Emergency services and event organizers who need traffic impact estimates

**Purpose**

Austin is one of the fastest-growing cities in the United States and consistently ranks among the worst cities for traffic congestion. Accurate short-term traffic prediction enables proactive signal control, smarter rerouting, and better event planning. Existing commercial systems rely heavily on historical averages and do not adapt well to unusual conditions like weather events, road closures, or major public gatherings.

**Context**

This project uses real sensor data from the City of Austin's radar traffic monitoring network, combined with historical weather data and a city-permitted events calendar. We are evaluating whether adding contextual modalities (weather, events) improves prediction accuracy over a sensor-only baseline, and building toward a full graph neural network model that captures spatial relationships between sensors across the road network.

---

## SLIDE 2 — Dataset

**Source 1 — Traffic Sensors (Primary)**

- Provider: City of Austin Open Data Portal (SODA API, dataset i626-g7ub)
- Sensor type: Radar sensors mounted at intersections, measuring vehicles in real time
- Coverage: 10 active sensors across Austin (5 others dropped due to >50% missing data)
- Time range: April 2020 – September 2021 (18 months)
- Resolution: Originally 15-minute intervals, upsampled to 5-minute resolution
- Features per sensor: speed (mph), volume (vehicle count), occupancy (% road utilization)
- Total size: 157,824 timesteps × 10 sensors × 3 features

**Source 2 — Weather Data**

- Provider: Open-Meteo historical weather API
- Resolution: Hourly, forward-filled to 5-minute resolution
- Features: Temperature (°F), precipitation (inches), wind speed (mph), humidity (%), visibility, weather code
- Why included: Rain, fog, and extreme temperatures directly affect driving speed and behavior

**Source 3 — Austin Events Data**

- Provider: City of Austin ACE (Austin Center for Events) Permits API
- Coverage: 31 real permitted public events with exact start/end timestamps
- Features: event active flag, number of concurrent events, road closure indicator, road segment closure count
- Challenge: COVID-19 eliminated virtually all permitted events from April–December 2020

**Data Split**

| Split      | Period                    | Timesteps | Windows  |
|------------|---------------------------|-----------|----------|
| Train      | Apr 2020 – Mar 2021       | 105,120   | 105,109  |
| Validation | Apr 2021 – Jun 2021       | 26,208    | 26,197   |
| Test       | Jul 2021 – Sep 2021       | 26,496    | 26,473   |

**Sliding Window Protocol**

Each training sample is a window of T=12 input timesteps (60 minutes of history) followed by H=12 target timesteps (60 minutes of future). The model is evaluated at three specific forecast horizons: 15 minutes, 30 minutes, and 60 minutes.

---

## SLIDE 3 — Proposed Method and Deep Learning Techniques

**Model 1 — ARIMA(2,1,2) [Classical Baseline]**

Autoregressive Integrated Moving Average model — the standard classical baseline in traffic forecasting literature. Fitted independently per sensor on 105,120 training timesteps. Uses speed history only; cannot incorporate weather or event context. Parameters: p=2 (autoregression), d=1 (differencing for stationarity), q=2 (moving average). Evaluation uses rolling H-step forecasts on all 26,473 test windows.

**Model 2 — LSTM Baseline (Sensor Data Only)**

- Architecture: 2-layer encoder-decoder LSTM, 64 hidden units, dropout 0.1
- Each sensor processed independently — no spatial cross-sensor information
- Encoder: reads 12 timesteps of (speed, volume, occupancy) → compressed hidden state
- Decoder: generates 12 future speed predictions step-by-step using teacher forcing during training
- Deep learning technique: Long Short-Term Memory (LSTM) encoder-decoder with teacher forcing
- Parameters: 101,441

**Teacher Forcing and Train–Test Mismatch**

Teacher forcing is a training technique for sequence decoders. At each decoder step, the model needs a "previous speed" as input to predict the next step. During training we use a teacher forcing ratio of 0.5: at each step, a coin flip decides whether to feed the ground-truth speed (teacher forcing) or the model's own previous prediction as the next input. This gives the model stable gradients early in training and helps it learn the correct output scale.

At inference (validation and test), the teacher forcing ratio is set to 0.0 — the decoder is fully autoregressive: each step's prediction becomes the next step's input. This is a deliberate and standard design choice, but it introduces a train–test mismatch known as *exposure bias*: the model trained partly on clean ground-truth inputs but is evaluated entirely on its own (potentially noisy) predictions. Errors can compound across the 12 decoder steps, which is one reason MAE grows from @15min to @60min faster than a simple per-step error rate would suggest. A future mitigation is scheduled sampling (gradually reducing the teacher forcing ratio to 0 over training), which was not implemented in this baseline.

**Model 3 — LSTM + Weather Context**

- Same encoder-decoder LSTM backbone as Model 2
- Adds a context encoder: MLP that compresses 12 timesteps × 6 weather features → 64-dimensional embedding
- Context injection: weather embedding projected and added to the LSTM decoder's initial hidden state
- Deep learning technique: MLP context encoder with hidden state injection
- Parameters: 139,969

**Model 4 — LSTM + Weather + Events Context**

- Identical to Model 3 but context includes 17 features per timestep:
  - 6 weather features
  - 7 calendar features (cyclic hour/day/month encodings, weekend flag)
  - 4 event features from real City of Austin permit data
- Deep learning technique: Cyclic feature encoding for calendar variables; multi-modal MLP fusion
- Parameters: 139,969

**Training Protocol (all deep learning models)**

- Optimizer: AdamW, weight decay 1e-4
- Learning rate: 1e-3 with ReduceLROnPlateau (factor 0.5, patience 5 epochs)
- Early stopping: patience 10 epochs on validation MAE
- Loss function: Huber loss (δ=1.0, robust to speed outliers)
- Hardware: Apple Silicon MPS (GPU-accelerated)
- Batch size: 64

---

## SLIDE 4 — Progress So Far and Results

**Quantitative Results — Test Set Performance**

| Model                              | MAE @15min | MAE @30min | MAE @60min | RMSE @15min | MAPE @15min |
|------------------------------------|------------|------------|------------|-------------|-------------|
| ARIMA(2,1,2)                       | 3.276 mph  | 3.542 mph  | 3.914 mph  | 8.501       | 7.84%       |
| Chronos-T5-Base (zero-shot)        | 3.279 mph  | 3.655 mph  | 3.977 mph  | 8.196       | 8.24%       |
| LSTM — Sensor Only                 | **3.174 mph**  | **3.372 mph**  | **3.637 mph**  | **7.096**   | **7.38%**   |
| LSTM + Weather                     | 3.227 mph  | 3.426 mph  | 3.701 mph  | 7.135       | 7.57%       |
| LSTM + Weather + Events            | 3.317 mph  | 3.503 mph  | 3.733 mph  | 7.202       | 7.84%       |

**Key Findings**

1. The LSTM baseline outperforms ARIMA by ~0.10 mph at 15 minutes, and the gap widens to ~0.28 mph at 60 minutes — confirming deep learning captures longer-range temporal patterns that ARIMA cannot.

2. Chronos-T5-Base (200M parameters, zero-shot, never trained on Austin data) matches ARIMA almost exactly: 3.279 vs 3.276 MAE at 15 minutes. Both are clearly beaten by our locally-trained LSTM. This shows that a general-purpose foundation model without fine-tuning cannot replace a domain-specific model trained on local patterns.

3. ARIMA's RMSE is significantly higher (8.50 vs 7.10) despite similar MAE. This means ARIMA makes larger errors during sudden speed changes — exactly the events a traffic system most needs to predict.

4. Adding weather data slightly hurts performance (+0.05 mph MAE). The simple hidden-state injection is too blunt to filter relevant weather signals from irrelevant ones.

5. Adding events data further hurts performance (+0.09 mph). Events coverage in 2020 is near zero due to COVID, giving the model too few examples to learn from.

6. MAE degrades faster with horizon than simple error accumulation predicts — a consequence of the train–test mismatch from teacher forcing (see model description above). The model trains with 50% ground-truth decoder inputs but is evaluated with 0%, so compounded prediction errors grow across the 12-step horizon.

**Work Completed (~40% of total project)**

- Full data pipeline: sensor download, weather download, events API integration, preprocessing, normalization, train/val/test splits
- 4 models fully implemented and evaluated: ARIMA, LSTM, LSTM+Weather, LSTM+Weather+Events
- Events data upgraded from manual curation to real City of Austin permit API with exact datetimes
- All results logged to structured JSON files; codebase on GitHub

---

## SLIDE 5 — Future Plans and Challenges

**Future Plans**

*Improve Context Fusion*

The current context encoder (MLP → hidden state injection) adds weather and event embeddings unconditionally to the LSTM decoder's initial hidden state. This is too coarse: a rainstorm at 2am has a different effect than a rainstorm during rush hour, but the model cannot make that distinction. A more principled approach would use attention-based fusion — letting each decoder step query which parts of the context window are most relevant at that moment.

*Address Exposure Bias with Scheduled Sampling*

The train–test mismatch from teacher forcing causes error compounding at longer horizons. Scheduled sampling — gradually reducing the teacher forcing ratio from 1.0 to 0.0 over the course of training — bridges this gap. The model would progressively learn to handle its own prediction errors rather than seeing clean ground-truth inputs throughout training.

*Fine-tune Chronos on Local Data*

Chronos-T5-Base as a zero-shot baseline matches ARIMA, showing that a foundation model pre-trained on general time series cannot capture Austin-specific traffic patterns without adaptation. Fine-tuning Chronos on the 18-month Austin training set (105,120 timesteps × 10 sensors) could demonstrate whether a foundation model can surpass domain-specific LSTM baselines when given local context.

**Challenges**

*1. Event Data Coverage Gap*

The City of Austin ACE Events API only covers city-permitted street events. Major traffic-generating events — UT Football at Darrell K Royal Stadium, Formula 1 at Circuit of the Americas, Austin FC at Q2 Stadium — occur at private venues outside the city permitting system and are absent from the dataset. Additionally, COVID-19 cancelled essentially all permitted events from April to December 2020, leaving 9 of our 18 training months with no event signal at all. This severely limits what the model can learn from the event features.

*2. Temporal Data Quality*

The raw sensor data had significant gaps — 5 of the original 15 sensors were dropped for more than 50% missing readings. January–March 2020 and October–December 2021 had to be excluded entirely after all sensors showed statistically imputed values (speed std ≈ 0.69 vs normal ≈ 8–10 mph), which would have produced artificially low test error.

*3. Context Injection vs. Attention*

Simply injecting weather and event features into the LSTM decoder hidden state at t=0 does not improve performance — the model cannot learn to selectively use context at individual decoder steps. Fixing this requires architectural changes beyond the LSTM baseline scope.
