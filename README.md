# ☕ Bean Counter — Specialty Coffee Expansion & Unit-Economics Simulator

A live, interactive model for deciding how a specialty coffee business should
grow: signing B2B wholesale accounts (cafes, offices, restaurants) vs. opening
company-owned retail cafes. Built to demonstrate the same unit-economics
fluency — CAC, ACV, churn, LTV:CAC, payback, TAM/SAM/SOM — that any B2B SaaS
or consumer-retail startup runs its growth decisions on, grounded in a domain
I've spent years studying independently.

**Live app:** _add your Streamlit Cloud URL here once deployed_
**Built by:** Ashlyn Chi Garcia — [LinkedIn](https://linkedin.com/in/ashlyn-chi) · [GitHub](https://github.com/ashlynchi14)

[![tests](https://github.com/ashlynchi14/bean-counter/actions/workflows/tests.yml/badge.svg)](https://github.com/ashlynchi14/bean-counter/actions/workflows/tests.yml)
_(badge goes live automatically once this is pushed to GitHub and the workflow runs once)_

## What it does

- Adjust wholesale channel assumptions (ACV, CAC, gross margin, churn, sales
  cycle, new-account velocity) and watch CAC payback, LTV:CAC, a 36-month
  3-scenario MRR projection, and a cohort retention curve update live.
- Adjust retail assumptions (build-out cost, rent, ticket size, traffic,
  COGS, labor) and see per-store payback and a cumulative cash-position
  curve as new cafes open.
- Pick a target region and see TAM → SAM → SOM funnel sizing.
- Read an auto-generated, threshold-driven executive takeaway — the same
  15-second read a Chief of Staff or founder would want — that changes as
  the recommendation changes.

## Why this project

Static resumes and GitHub repos full of `.ipynb` notebooks don't show how
someone thinks under pressure with a founder watching. This does: it's a
link, not a file — someone can open it, drag a slider, and see the
model's logic (and its limits) in real time.

The underlying assumptions come from specialty coffee industry norms,
informed by multi-year independent research (20+ industry contacts, 5
producing regions, 10+ cities visited) — not a domain borrowed for the
occasion. The model structure — CAC/ACV/churn/payback/TAM-SAM-SOM — is the
general-purpose BizOps toolkit and translates directly to any B2B or
consumer-retail business.

## Run locally

```bash
git clone https://github.com/ashlynchi14/bean-counter.git
cd bean-counter
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Deploy for free — GitHub + Streamlit Community Cloud

1. **Push to GitHub.**
   ```bash
   cd bean-counter
   git init
   git add .
   git commit -m "Initial commit: Bean Counter expansion simulator"
   git branch -M main
   git remote add origin https://github.com/ashlynchi14/bean-counter.git
   git push -u origin main
   ```
   (Create the empty `bean-counter` repo on GitHub first if it doesn't exist
   yet — github.com/new.)

2. **Deploy on Streamlit Community Cloud.**
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
     your GitHub account.
   - Click **"New app"**, pick the `bean-counter` repo, branch `main`, and
     main file path `app.py`.
   - Click **Deploy**. First build takes 1–2 minutes; you'll get a URL like
     `https://bean-counter-ashlynchi14.streamlit.app`.

3. **Pin the link.**
   - Add the live URL to the top of your resume next to your GitHub and
     LinkedIn.
   - Add it to your LinkedIn "Featured" section.
   - Drop it directly in outreach — see the message template below.

## Testing

`model.py` is pure pandas/NumPy with no UI code, so it's fully unit-tested —
18 `pytest` cases in `tests/test_model.py` covering edge cases (zero churn,
zero margin, unprofitable retail assumptions), monotonicity (higher ACV pays
back faster, lower churn improves LTV:CAC), and invariants (SOM can never
exceed SAM). This is optional for running the app, so it's kept out of the
main `requirements.txt` to keep the Streamlit Cloud deploy lean.

```bash
pip install -r requirements-dev.txt
pytest
```

A GitHub Actions workflow (`.github/workflows/tests.yml`) runs this suite on
every push, so the repo shows a passing-tests badge instead of just claiming
the tests exist.

## Outreach templates

**Short version** (cold LinkedIn/email opener):

> Hey [Name] — I built a live model of expansion economics for a specialty
> coffee business (wholesale accounts vs. retail cafes — CAC payback,
> LTV:CAC, TAM/SAM/SOM). Same toolkit I'd bring to [Company]'s growth
> decisions. Takes 15 seconds to play with: [URL]

**Longer version** (email or InMail to a Chief of Staff / VP Ops / founder):

> Subject: CMU Tepper Strategy grad, with a live build
>
> Hi [Name],
>
> I saw [Company] is scaling [growth motion / expansion effort] — as you grow,
> keeping unit economics tight across channels gets harder fast.
>
> I'm a CMU Tepper MS grad (Strategy Track) with a biotechnology engineering
> background, and I build operational frameworks from scratch. To show that
> concretely, I built and open-sourced a live interactive simulator called
> Bean Counter that models a B2B wholesale vs. retail expansion decision
> using SaaS-style unit economics — LTV:CAC, churn, payback curves, TAM/SAM/SOM:
> [Your Streamlit Live URL]
>
> Tests included, code's public: [GitHub URL]
>
> Would love to bring that kind of data-driven execution to [Company]. Open to
> a quick call this week?
>
> Best,
> Ashlyn Chi Garcia
> github.com/ashlynchi14 · linkedin.com/in/ashlyn-chi

Swap "[growth motion / expansion effort]" for something specific to the
company you're messaging — a generic line here is the one part of this
outreach a reader will notice is templated.

## Project structure

```
bean-counter/
├── app.py                      # Streamlit UI — layout, charts, KPI cards
├── model.py                    # Pure pandas/NumPy logic — no UI code
├── tests/
│   └── test_model.py           # 18 pytest cases against model.py
├── .github/workflows/
│   └── tests.yml               # runs pytest on every push
├── requirements.txt            # runtime deps only (what Streamlit Cloud installs)
├── requirements-dev.txt        # + pytest, for local test runs
├── .streamlit/
│   └── config.toml             # theme
└── README.md
```

`model.py` is kept UI-free on purpose — it's the part worth reading if
someone wants to check your math, and it's the part the tests exercise.

## Extending it

- Swap in a real dataset (e.g., NYC Open Data business listings) for the
  TAM/SAM region defaults instead of the illustrative placeholders in
  `model.py::REGION_DEFAULTS`.
- Add a "download scenario as PDF" button (`st.download_button` +
  a simple report generator) if you want a leave-behind after a call.
