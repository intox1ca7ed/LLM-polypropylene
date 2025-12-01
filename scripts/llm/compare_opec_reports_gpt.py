import os
import time
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()   # Load .env file with OPENAI_API_KEY

# -------- CONFIG --------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "reports" / "energy" / "opec"
IN_FILE = DATA_DIR / "opec_texts_clean_sections.csv"
OUT_FILE = DATA_DIR / "opec_comparison_scores_gpt.csv"

FOCUS_SECTIONS = {
    "World Oil Demand",
    "World Oil Supply",
    "Balance of Supply and Demand",
    "Crude and Product Prices",
}

MODEL_NAME = "gpt-4o-mini"
MAX_CHARS = 4500
RETRIES = 3

def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)


def assemble_text(df_report):
    """Combine only key sections for better signal."""
    parts = []
    for _, r in df_report.iterrows():
        if r["section"] in FOCUS_SECTIONS:
            parts.append(f"### {r['section']}\n{str(r['content']).strip()}")
    if not parts:
        parts = df_report["content"].astype(str).tolist()
    full_text = "\n\n".join(parts)
    return full_text[:MAX_CHARS]


def build_prompt(prev_meta, prev_text, curr_meta, curr_text):
    return f"""
You are an expert energy market analyst.

Compare the tone and outlook between the two OPEC Monthly Oil Market Reports below.

Focus ONLY on:
- Global oil demand outlook
- Oil supply (OPEC + non-OPEC)
- Crude oil price expectations

Report A (Previous): {prev_meta}
---
{prev_text}
---

Report B (Current): {curr_meta}
---
{curr_text}
---

Return ONLY a JSON object:
{{
  "comparison_score": <number between -1 and 1>,
  "tone_change": "more bullish | more bearish | neutral",
  "summary": "<brief explanation>"
}}
""".strip()


def call_gpt(client, prompt):
    for attempt in range(RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = resp.choices[0].message.content.strip()

            # strip code fences if present
            if content.startswith("```"):
                content = content.strip("`")
                if content.lower().startswith("json"):
                    content = content[4:].strip()

            return json.loads(content)

        except Exception as e:
            if attempt == RETRIES - 1:
                return {
                    "comparison_score": None,
                    "tone_change": None,
                    "summary": f"ERROR: {str(e)}"
                }
            time.sleep(2)

def main():
    client = get_client()

    df = pd.read_csv(IN_FILE)
    df["report_date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month_name"].astype(str) + "-01",
        errors="coerce"
    )

    grouped = df.sort_values("report_date").groupby("report_date")

    reports = []
    for rep_date, g in grouped:
        meta = f"{g.iloc[0]['month_name']} {g.iloc[0]['year']}"
        text = assemble_text(g)
        reports.append({"date": rep_date, "meta": meta, "text": text})

    results = []

    print(" Starting GPT comparison across reports...\n")

    for i in tqdm(range(1, len(reports))):
        prev_r = reports[i - 1]
        curr_r = reports[i]

        prompt = build_prompt(prev_r["meta"], prev_r["text"], curr_r["meta"], curr_r["text"])
        result = call_gpt(client, prompt)

        results.append({
            "date": curr_r["date"],
            "year": curr_r["date"].year,
            "month": curr_r["date"].month,
            "month_name": curr_r["meta"].split()[0],
            "comparison_score": result.get("comparison_score"),
            "tone_change": result.get("tone_change"),
            "summary": result.get("summary"),
            "prev_date": prev_r["date"],
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_FILE, index=False)
    print(f"\n GPT comparison scores saved to: {OUT_FILE}")


if __name__ == "__main__":
    main()
