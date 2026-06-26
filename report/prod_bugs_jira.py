"""
prod_bugs_jira.py
-----------------
Production Bug Report (Jira) â€” Generator

Pipeline:
  1. Fetch all open production bugs from Jira project PPPB via REST API
     (replicates the "All Open Prod bugs" saved filter on board 6973).
  2. Compute metrics: severity, security vs non-security, aging, trends.
  3. Generate an AI executive summary via Azure OpenAI.
  4. Render an HTML report and save to the output path.

Credentials are read from the central dashboard .env file:
  CONFLUENCE_EMAIL       Jira/Atlassian account email
  CONFLUENCE_API_TOKEN   Jira/Atlassian API token
  AZURE_OPENAI_KEY       Azure OpenAI API key
  AZURE_OPENAI_ENDPOINT  Azure OpenAI resource endpoint URL
  AZURE_OPENAI_DEPLOY    Azure OpenAI deployment name
"""

import os
import json
import datetime
import base64
import re
import math
from collections import defaultdict
from io import BytesIO

import requests
from requests.auth import HTTPBasicAuth
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load credentials
# ---------------------------------------------------------------------------
# Credentials from environment variables (GitHub Secrets)

JIRA_BASE_URL    = "https://veradigm.atlassian.net"
JIRA_EMAIL       = os.environ.get("CONFLUENCE_EMAIL", "")
JIRA_TOKEN       = os.environ.get("CONFLUENCE_API_TOKEN", "")
JIRA_PROJECT     = "PPPB"

# Confluence publishing â€” page is created/updated automatically after each run
CONFLUENCE_SPACE_KEY   = os.environ.get("CONFLUENCE_SPACE_KEY", "PAY")  # Veradigm Payer space
CONFLUENCE_PAGE_TITLE  = "PPPB Production Bug Report"
CONFLUENCE_PAGE_ID_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "confluence_page_id.txt"
)

AZURE_OPENAI_KEY      = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOY   = os.environ.get("AZURE_OPENAI_DEPLOY", "gpt-4.1-mini")

OUTPUT_PATH = os.path.join(
    r"C:\Users\PeterLobo\OneDrive - Veradigm Corporate",
    "Peter", "AI_Agent_Reports", "production_stability_report_jira.html",
)

PREVIOUS_COUNTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "previous_counts_jira.json",
)

# Jira fields
SEVERITY_FIELD    = "priority"   # Jira priority maps to severity: Highest/High/Medium/Low/Lowest
SECURITY_LABEL    = "security"   # bugs labelled "security" or "Security" are security bugs
SECURITY_COMPONENT = "security"  # or component name containing "security"

# States treated as "closed"
CLOSED_STATUSES = {"Done", "Closed", "Resolved", "Won't Fix", "Duplicate"}

# ---------------------------------------------------------------------------
# SLA policy (director announcement 2026-06-26)
# Applies to NEW bugs only â€” created on/after SLA_POLICY_START.
# Clock starts when a bug first moves to "In Progress".
# ---------------------------------------------------------------------------
SLA_POLICY_START = datetime.date(2026, 6, 26)
SLA_DAYS = {"Critical": 3, "High": 14, "Medium": 21, "Low": 35}
SLA_CLOCK_STATUS = "In Progress"

# Pre-triage statuses â€” SLA clock has NOT started yet
PRE_TRIAGE_STATUSES = {"Needs Triage", "Open", "In Planning", "In Icebox", "Needs Information"}

# ---------------------------------------------------------------------------
# Jira API helpers
# ---------------------------------------------------------------------------

AUTH = HTTPBasicAuth(JIRA_EMAIL, JIRA_TOKEN) if JIRA_EMAIL and JIRA_TOKEN else None

def _jira_get(path: str, params: dict = None):
    url = f"{JIRA_BASE_URL}/rest/api/3{path}"
    resp = requests.get(url, auth=AUTH, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _jira_search_page(jql: str, page_size: int, fields: list, next_page_token: str = None) -> dict:
    """POST to /rest/api/3/search/jql using cursor-based pagination."""
    url = f"{JIRA_BASE_URL}/rest/api/3/search/jql"
    payload = {
        "jql": jql,
        "maxResults": page_size,
        "fields": fields,
    }
    if next_page_token:
        payload["nextPageToken"] = next_page_token
    resp = requests.post(url, auth=AUTH, json=payload, timeout=30)
    if not resp.ok:
        print(f"  Jira search error {resp.status_code}: {resp.text[:500]}")
    resp.raise_for_status()
    return resp.json()


def fetch_all_bugs(jql: str) -> list:
    """Paginate through all Jira issues matching jql, return list of issue dicts."""
    page_size = 100
    issues = []
    fields = ["summary", "status", "priority", "labels", "components",
              "created", "resolutiondate", "updated", "issuetype", "assignee", "reporter"]
    next_token = None
    while True:
        data = _jira_search_page(jql, page_size, fields, next_token)
        batch = data.get("issues", [])
        issues.extend(batch)
        next_token = data.get("nextPageToken")
        if not next_token or len(batch) < page_size:
            break
    return issues


def fetch_in_progress_date(issue_key: str) -> datetime.date | None:
    """Return the date this issue first transitioned to 'In Progress' via changelog."""
    start_at = 0
    while True:
        data = _jira_get(f"/issue/{issue_key}/changelog", params={"startAt": start_at, "maxResults": 100})
        for entry in data.get("values", []):
            for item in entry.get("items", []):
                if item.get("field") == "status" and item.get("toString") == SLA_CLOCK_STATUS:
                    return parse_date(entry.get("created", ""))
        if data.get("isLast", True):
            break
        start_at += len(data.get("values", []))
    return None


def fetch_sla_start_dates(bugs: list) -> dict:
    """
    For each bug that is past pre-triage, fetch its first In Progress date.
    Returns {issue_key: date_or_None}.
    """
    result = {}
    eligible = [
        b for b in bugs
        if parse_date(b["fields"].get("created", "")) is not None
        and parse_date(b["fields"].get("created", "")) >= SLA_POLICY_START
        and (b["fields"].get("status") or {}).get("name", "") not in PRE_TRIAGE_STATUSES
    ]
    print(f"  Fetching SLA changelogs for {len(eligible)} eligible bugsâ€¦")
    for i, bug in enumerate(eligible, 1):
        key = bug["key"]
        result[key] = fetch_in_progress_date(key)
        if i % 10 == 0:
            print(f"    {i}/{len(eligible)} doneâ€¦")
    return result


def fetch_recent_closed_bugs(days: int = 180) -> list:
    """Fetch bugs closed in the last `days` days (for trend charts)."""
    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    jql = (
        f'project = {JIRA_PROJECT} AND issuetype = Bug '
        f'AND statusCategory = Done AND updated >= "{cutoff}"'
    )
    return fetch_all_bugs(jql)


def fetch_recently_opened_bugs(days: int = 180) -> list:
    """Fetch bugs created in the last `days` days."""
    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    jql = (
        f'project = {JIRA_PROJECT} AND issuetype = Bug '
        f'AND created >= "{cutoff}"'
    )
    return fetch_all_bugs(jql)


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

PRIORITY_ORDER = ["Highest", "High", "Medium", "Low", "Lowest"]
PRIORITY_LABEL_MAP = {
    "Highest": "Critical",
    "High":    "High",
    "Medium":  "Medium",
    "Low":     "Low",
    "Lowest":  "Low",
}

def get_severity(issue: dict) -> str:
    priority = (issue["fields"].get("priority") or {}).get("name", "Medium")
    return PRIORITY_LABEL_MAP.get(priority, "Medium")


def is_security(issue: dict) -> bool:
    labels = [l.lower() for l in (issue["fields"].get("labels") or [])]
    if SECURITY_LABEL in labels:
        return True
    components = [(c.get("name") or "").lower() for c in (issue["fields"].get("components") or [])]
    if any(SECURITY_COMPONENT in c for c in components):
        return True
    summary = (issue["fields"].get("summary") or "").lower()
    if "security" in summary or "vulnerability" in summary or "cve" in summary:
        return True
    return False


def parse_date(date_str: str) -> datetime.date | None:
    if not date_str:
        return None
    try:
        return datetime.datetime.fromisoformat(date_str[:10]).date()
    except Exception:
        return None


def age_days(created_str: str) -> int:
    d = parse_date(created_str)
    if d is None:
        return 0
    return (datetime.date.today() - d).days


# ---------------------------------------------------------------------------
# SLA helpers
# ---------------------------------------------------------------------------

def sla_status_for_bug(bug: dict, in_progress_date: datetime.date | None) -> dict:
    """
    Returns a dict describing the SLA status for a single bug.
    Keys: exempt, not_started, sla_days, elapsed_days, pct_used, status
    status is one of: 'on_track', 'at_risk', 'breached', 'not_started', 'exempt'
    """
    created = parse_date(bug["fields"].get("created", ""))
    sev = get_severity(bug)
    sla_days = SLA_DAYS.get(sev, 21)

    # Bugs created before policy date are backlog-exempt
    if created is None or created < SLA_POLICY_START:
        return {"status": "exempt", "sla_days": sla_days, "elapsed_days": None,
                "pct_used": None, "in_progress_date": None}

    # SLA clock hasn't started (still pre-triage)
    if in_progress_date is None:
        curr_status = (bug["fields"].get("status") or {}).get("name", "")
        if curr_status in PRE_TRIAGE_STATUSES:
            return {"status": "not_started", "sla_days": sla_days, "elapsed_days": None,
                    "pct_used": None, "in_progress_date": None}
        # Past triage but no In Progress entry found â€” use created as fallback
        in_progress_date = created

    elapsed = (datetime.date.today() - in_progress_date).days
    pct = round(elapsed / sla_days * 100)

    if pct > 100:
        status = "breached"
    elif pct >= 70:
        status = "at_risk"
    else:
        status = "on_track"

    return {
        "status": status,
        "sla_days": sla_days,
        "elapsed_days": elapsed,
        "pct_used": pct,
        "in_progress_date": in_progress_date.isoformat() if in_progress_date else None,
    }


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(open_bugs: list, closed_bugs: list, opened_bugs: list,
                    sla_start_dates: dict = None) -> dict:
    today = datetime.date.today()
    # --- open bug counts by severity & security ---
    severity_counts = defaultdict(int)
    sec_count = non_sec_count = 0
    aging_buckets = {">365d": 0, ">180d": 0, ">90d": 0, "<=90d": 0}

    for bug in open_bugs:
        sev = get_severity(bug)
        severity_counts[sev] += 1
        if is_security(bug):
            sec_count += 1
        else:
            non_sec_count += 1
        age = age_days(bug["fields"].get("created", ""))
        if age > 365:
            aging_buckets[">365d"] += 1
        elif age > 180:
            aging_buckets[">180d"] += 1
        elif age > 90:
            aging_buckets[">90d"] += 1
        else:
            aging_buckets["<=90d"] += 1

    # --- monthly open/close trend (last 6 months) ---
    months = []
    for i in range(5, -1, -1):
        d = today.replace(day=1) - datetime.timedelta(days=1)
        for _ in range(i):
            d = d.replace(day=1) - datetime.timedelta(days=1)
        months.append(d.replace(day=1))

    def month_key(d: datetime.date) -> str:
        return d.strftime("%Y-%m")

    opened_by_month = defaultdict(int)
    for bug in opened_bugs:
        d = parse_date(bug["fields"].get("created", ""))
        if d:
            opened_by_month[month_key(d)] += 1

    closed_by_month = defaultdict(int)
    for bug in closed_bugs:
        d = parse_date(bug["fields"].get("resolutiondate") or bug["fields"].get("updated", ""))
        if d:
            closed_by_month[month_key(d)] += 1

    trend = []
    for m in months:
        mk = month_key(m)
        trend.append({
            "month": m.strftime("%b %Y"),
            "opened": opened_by_month.get(mk, 0),
            "closed": closed_by_month.get(mk, 0),
        })

    # --- top assignees (open bugs) ---
    assignee_counts = defaultdict(int)
    for bug in open_bugs:
        a = bug["fields"].get("assignee")
        name = (a or {}).get("displayName", "Unassigned")
        assignee_counts[name] += 1
    top_assignees = sorted(assignee_counts.items(), key=lambda x: -x[1])[:10]

    # --- top issues list (most critical, oldest first within Critical) ---
    def sort_key(b):
        sev_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        s = get_severity(b)
        return (sev_order.get(s, 9), -age_days(b["fields"].get("created", "")))

    top_issues = sorted(open_bugs, key=sort_key)[:20]

    # --- SLA summary ---
    sla_start_dates = sla_start_dates or {}
    sla_summary = {"on_track": 0, "at_risk": 0, "breached": 0, "not_started": 0, "exempt": 0}
    sla_details = []   # list of dicts for the SLA table
    for bug in open_bugs:
        ip_date_str = sla_start_dates.get(bug["key"])
        ip_date = parse_date(ip_date_str) if ip_date_str else None
        sla = sla_status_for_bug(bug, ip_date)
        sla_summary[sla["status"]] = sla_summary.get(sla["status"], 0) + 1
        if sla["status"] not in ("exempt",):
            sla_details.append({"bug": bug, "sla": sla})

    # Sort SLA details: breached first, then at_risk, then not_started, then on_track
    _sla_order = {"breached": 0, "at_risk": 1, "not_started": 2, "on_track": 3}
    sla_details.sort(key=lambda x: (
        _sla_order.get(x["sla"]["status"], 9),
        -(x["sla"]["elapsed_days"] or 0)
    ))

    return {
        "total_open": len(open_bugs),
        "security_open": sec_count,
        "non_security_open": non_sec_count,
        "severity_counts": dict(severity_counts),
        "aging_buckets": aging_buckets,
        "trend": trend,
        "top_assignees": top_assignees,
        "top_issues": top_issues,
        "sla_summary": sla_summary,
        "sla_details": sla_details,
        "sla_policy_start": SLA_POLICY_START.isoformat(),
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Chart generators (return base64 PNG strings)
# ---------------------------------------------------------------------------

COLORS = {
    "Critical": "#e53e3e",
    "High":     "#dd6b20",
    "Medium":   "#d69e2e",
    "Low":      "#38a169",
    "opened":   "#667eea",
    "closed":   "#48bb78",
    "security": "#e53e3e",
    "non-sec":  "#4299e1",
}


def _b64_chart(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, transparent=False)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def chart_severity_donut(severity_counts: dict) -> str:
    labels = [k for k in ["Critical", "High", "Medium", "Low"] if severity_counts.get(k, 0) > 0]
    values = [severity_counts.get(k, 0) for k in labels]
    clrs   = [COLORS.get(k, "#a0aec0") for k in labels]
    fig, ax = plt.subplots(figsize=(4.2, 3.6), facecolor="#ffffff")
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=clrs, autopct="%1.0f%%",
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2),
    )
    for t in texts:
        t.set_fontsize(9)
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("Open Bugs by Severity", fontsize=10, fontweight="bold", pad=10)
    return _b64_chart(fig)


def chart_security_bar(sec: int, non_sec: int) -> str:
    fig, ax = plt.subplots(figsize=(4.2, 3.0), facecolor="#ffffff")
    bars = ax.bar(["Security", "Non-Security"], [sec, non_sec],
                  color=[COLORS["security"], COLORS["non-sec"]], width=0.45,
                  edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, [sec, non_sec]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Security vs Non-Security", fontsize=10, fontweight="bold")
    ax.set_ylabel("Open Bugs")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return _b64_chart(fig)


def chart_aging(aging_buckets: dict) -> str:
    labels = ["â‰¤90 days", ">90 days", ">180 days", ">365 days"]
    keys   = ["<=90d", ">90d", ">180d", ">365d"]
    values = [aging_buckets.get(k, 0) for k in keys]
    clrs   = ["#48bb78", "#d69e2e", "#dd6b20", "#e53e3e"]
    fig, ax = plt.subplots(figsize=(5.0, 3.0), facecolor="#ffffff")
    bars = ax.barh(labels, values, color=clrs, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=10, fontweight="bold")
    ax.set_title("Aging Distribution", fontsize=10, fontweight="bold")
    ax.set_xlabel("Open Bugs")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return _b64_chart(fig)


def chart_trend(trend: list) -> str:
    months  = [t["month"] for t in trend]
    opened  = [t["opened"] for t in trend]
    closed  = [t["closed"] for t in trend]
    x = range(len(months))
    fig, ax = plt.subplots(figsize=(6.5, 3.4), facecolor="#ffffff")
    ax.plot(list(x), opened, marker="o", color=COLORS["opened"], linewidth=2,
            markersize=6, label="Opened")
    ax.plot(list(x), closed, marker="s", color=COLORS["closed"], linewidth=2,
            markersize=6, label="Closed")
    ax.fill_between(list(x), opened, alpha=0.12, color=COLORS["opened"])
    ax.fill_between(list(x), closed, alpha=0.12, color=COLORS["closed"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(months, rotation=20, ha="right", fontsize=8)
    ax.set_title("Monthly Bug Opened vs Closed (last 6 months)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Bugs")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return _b64_chart(fig)


def chart_sla_summary(sla_summary: dict) -> str:
    labels = ["On Track", "At Risk", "Breached", "Not Started"]
    keys   = ["on_track", "at_risk", "breached", "not_started"]
    values = [sla_summary.get(k, 0) for k in keys]
    clrs   = ["#48bb78", "#d69e2e", "#e53e3e", "#a0aec0"]
    fig, ax = plt.subplots(figsize=(4.5, 3.0), facecolor="#ffffff")
    bars = ax.bar(labels, values, color=clrs, width=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("SLA Compliance (new bugs)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Bugs")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    return _b64_chart(fig)


# ---------------------------------------------------------------------------
# AI executive summary
# ---------------------------------------------------------------------------

def generate_ai_summary(metrics: dict) -> str:
    if not (AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT):
        return "<em>AI summary unavailable â€” AZURE_OPENAI_KEY or AZURE_OPENAI_ENDPOINT not configured.</em>"
    try:
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version="2024-02-01",
        )
        sla = metrics.get("sla_summary", {})
        prompt = f"""You are a software quality engineering manager writing a brief executive summary of the current production bug state for a Payerpath product dashboard.

Key metrics (from Jira project PPPB):
- Total open production bugs: {metrics['total_open']}
- Security bugs open: {metrics['security_open']}
- Non-security bugs open: {metrics['non_security_open']}
- Severity breakdown: {json.dumps(metrics['severity_counts'])}
- Aging: {json.dumps(metrics['aging_buckets'])}
- 6-month trend (last month shown last): {json.dumps(metrics['trend'])}
- New SLA compliance (bugs created after {metrics.get('sla_policy_start','2026-06-26')}):
    On Track: {sla.get('on_track',0)}, At Risk: {sla.get('at_risk',0)}, Breached: {sla.get('breached',0)}, Not Started (pre-triage): {sla.get('not_started',0)}

SLA targets: Critical=3 days, High=14 days, Medium=21 days, Low=35 days (clock starts at In Progress).

Write a 3-4 sentence executive summary highlighting:
1. Overall health and any concerning trends
2. Security posture (security bug count and severity)
3. SLA compliance status for new bugs
4. One actionable recommendation

Be concise and factual. Use plain text â€” no markdown, no bullet points."""

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOY,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"<em>AI summary error: {exc}</em>"


# ---------------------------------------------------------------------------
# MoM delta loading/saving
# ---------------------------------------------------------------------------

def load_previous_counts() -> dict:
    try:
        if os.path.isfile(PREVIOUS_COUNTS_FILE):
            with open(PREVIOUS_COUNTS_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_current_counts(metrics: dict):
    data = {
        "saved_at": metrics["generated_at"],
        "total_open": metrics["total_open"],
        "security_open": metrics["security_open"],
        "non_security_open": metrics["non_security_open"],
        "severity_counts": metrics["severity_counts"],
    }
    try:
        with open(PREVIOUS_COUNTS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def delta_str(current: int, previous: int | None) -> str:
    if previous is None:
        return ""
    diff = current - previous
    if diff > 0:
        return f'<span class="delta-up">â–² {diff}</span>'
    if diff < 0:
        return f'<span class="delta-down">â–¼ {abs(diff)}</span>'
    return '<span class="delta-flat">â€” no change</span>'


# ---------------------------------------------------------------------------
# HTML report renderer
# ---------------------------------------------------------------------------

SEVERITY_BADGE_COLORS = {
    "Critical": "#e53e3e",
    "High":     "#dd6b20",
    "Medium":   "#d69e2e",
    "Low":      "#38a169",
}

STATUS_COLORS = {
    "Open":        "#4299e1",
    "In Progress":  "#9f7aea",
    "Reopened":    "#f6ad55",
    "Done":        "#48bb78",
    "Closed":      "#a0aec0",
    "Resolved":    "#68d391",
}


def render_html(metrics: dict, charts: dict, ai_summary: str, prev: dict) -> str:
    generated_at = metrics["generated_at"]
    total        = metrics["total_open"]
    sec          = metrics["security_open"]
    non_sec      = metrics["non_security_open"]
    sev          = metrics["severity_counts"]
    aging        = metrics["aging_buckets"]
    top_issues   = metrics["top_issues"]

    crit = sev.get("Critical", 0)
    high = sev.get("High", 0)
    med  = sev.get("Medium", 0)
    low  = sev.get("Low", 0)

    d_total  = delta_str(total, prev.get("total_open"))
    d_sec    = delta_str(sec, prev.get("security_open"))
    d_crit   = delta_str(crit, (prev.get("severity_counts") or {}).get("Critical"))
    d_high   = delta_str(high, (prev.get("severity_counts") or {}).get("High"))

    def img(b64, alt=""):
        return f'<img src="data:image/png;base64,{b64}" alt="{alt}" class="chart-img"/>'

    # Top issues table rows
    rows = []
    for bug in top_issues:
        f    = bug["fields"]
        key  = bug["key"]
        url  = f"{JIRA_BASE_URL}/browse/{key}"
        sev_val = get_severity(bug)
        badge_color = SEVERITY_BADGE_COLORS.get(sev_val, "#a0aec0")
        status = (f.get("status") or {}).get("name", "Open")
        sc = STATUS_COLORS.get(status, "#a0aec0")
        age = age_days(f.get("created", ""))
        summary = (f.get("summary") or "")[:80]
        sec_flag = "ðŸ”’" if is_security(bug) else ""
        assignee_info = f.get("assignee")
        assignee = (assignee_info or {}).get("displayName", "Unassigned") if assignee_info else "Unassigned"
        rows.append(f"""
          <tr>
            <td><a href="{url}" target="_blank" class="issue-link">{key}</a></td>
            <td><span class="sev-badge" style="background:{badge_color}">{sev_val}</span></td>
            <td class="issue-summary">{sec_flag} {summary}</td>
            <td><span class="status-badge" style="background:{sc}">{status}</span></td>
            <td>{age}d</td>
            <td>{assignee}</td>
          </tr>""")

    rows_html = "".join(rows)

    # SLA detail table rows
    sla_rows = []
    for entry in metrics.get("sla_details", []):
        bug   = entry["bug"]
        sla   = entry["sla"]
        f     = bug["fields"]
        key   = bug["key"]
        url   = f"{JIRA_BASE_URL}/browse/{key}"
        sev_val = get_severity(bug)
        badge_color = SEVERITY_BADGE_COLORS.get(sev_val, "#a0aec0")
        status_name = (f.get("status") or {}).get("name", "Open")
        sc          = STATUS_COLORS.get(status_name, "#a0aec0")
        summary     = (f.get("summary") or "")[:70]
        assignee_i  = f.get("assignee")
        assignee    = (assignee_i or {}).get("displayName", "Unassigned") if assignee_i else "Unassigned"

        sla_st  = sla["status"]
        sla_cls = sla_st.replace("_", "-")
        sla_lbl = sla_st.replace("_", " ").title()
        pct     = sla.get("pct_used")
        elapsed = sla.get("elapsed_days")
        sla_days = sla.get("sla_days", "â€”")

        if sla_st == "not_started":
            bar_html  = "â€”"
            pct_html  = "â€”"
            elapsed_html = "â€”"
        else:
            bar_pct  = min(pct, 100) if pct is not None else 0
            bar_color = {"on_track": "#48bb78", "at_risk": "#d69e2e", "breached": "#e53e3e"}.get(sla_st, "#a0aec0")
            bar_html  = f'<div class="sla-bar-wrap"><div class="sla-bar" style="width:{bar_pct}%;background:{bar_color}"></div></div>'
            pct_html  = f"{pct}%" if pct is not None else "â€”"
            elapsed_html = f"{elapsed}d" if elapsed is not None else "â€”"

        ip_date = sla.get("in_progress_date", "â€”") or "â€”"

        sla_rows.append(f"""
          <tr>
            <td><a href="{url}" target="_blank" class="issue-link">{key}</a></td>
            <td><span class="sev-badge" style="background:{badge_color}">{sev_val}</span></td>
            <td class="issue-summary">{summary}</td>
            <td><span class="status-badge" style="background:{sc}">{status_name}</span></td>
            <td>{ip_date}</td>
            <td>{elapsed_html} / {sla_days}d</td>
            <td>{bar_html}</td>
            <td style="font-weight:700">{pct_html}</td>
            <td><span class="sla-badge {sla_cls}">{sla_lbl}</span></td>
            <td>{assignee}</td>
          </tr>""")
    sla_rows_html = "".join(sla_rows) if sla_rows else '<tr><td colspan="10" style="text-align:center;color:#a0aec0;padding:20px">No new bugs under SLA tracking yet</td></tr>'

    # Assignee table
    assignee_rows = ""
    for name, cnt in metrics["top_assignees"]:
        pct = round(cnt / total * 100) if total else 0
        assignee_rows += f"""
          <tr>
            <td>{name}</td>
            <td><div class="mini-bar-wrap"><div class="mini-bar" style="width:{pct}%;background:#667eea"></div></div></td>
            <td style="text-align:right;font-weight:600">{cnt}</td>
          </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Production Bug Report â€” Jira</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #f0f4f8;
    color: #2d3748;
    padding: 20px;
  }}
  .page-header {{
    background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
    color: white;
    padding: 24px 28px;
    border-radius: 12px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .page-header h1 {{ font-size: 1.5rem; font-weight: 700; }}
  .page-header .subtitle {{ font-size: 0.85rem; opacity: 0.7; margin-top: 4px; }}
  .header-meta {{ text-align: right; font-size: 0.8rem; opacity: 0.8; }}
  .header-links {{ margin-top: 8px; }}
  .header-links a {{
    color: #90cdf4;
    text-decoration: none;
    font-size: 0.78rem;
    margin-left: 12px;
  }}
  .header-links a:first-child {{ margin-left: 0; }}
  .migration-note {{
    background: #fff3cd;
    border-left: 4px solid #f6ad55;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 20px;
    font-size: 0.85rem;
    color: #744210;
  }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 14px;
    margin-bottom: 20px;
  }}
  .kpi-card {{
    background: white;
    border-radius: 10px;
    padding: 18px 16px;
    box-shadow: 0 1px 8px rgba(0,0,0,.07);
    text-align: center;
  }}
  .kpi-label {{ font-size: 0.72rem; color: #718096; text-transform: uppercase; letter-spacing: .05em; font-weight: 600; margin-bottom: 6px; }}
  .kpi-value {{ font-size: 2rem; font-weight: 800; color: #2d3748; line-height: 1; }}
  .kpi-delta {{ font-size: 0.78rem; margin-top: 6px; color: #718096; min-height: 1.2em; }}
  .delta-up   {{ color: #e53e3e; font-weight: 600; }}
  .delta-down {{ color: #38a169; font-weight: 600; }}
  .delta-flat {{ color: #718096; }}
  .kpi-card.critical .kpi-value {{ color: #e53e3e; }}
  .kpi-card.high     .kpi-value {{ color: #dd6b20; }}
  .kpi-card.medium   .kpi-value {{ color: #d69e2e; }}
  .kpi-card.security .kpi-value {{ color: #e53e3e; }}
  .section {{
    background: white;
    border-radius: 10px;
    padding: 20px 22px;
    margin-bottom: 20px;
    box-shadow: 0 1px 8px rgba(0,0,0,.07);
  }}
  .section-title {{
    font-size: 0.95rem;
    font-weight: 700;
    color: #2d3748;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 2px solid #edf2f7;
  }}
  .charts-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
  }}
  .chart-section {{
    background: white;
    border-radius: 10px;
    padding: 18px;
    box-shadow: 0 1px 8px rgba(0,0,0,.07);
    text-align: center;
  }}
  .chart-img {{ max-width: 100%; height: auto; border-radius: 6px; }}
  .ai-summary {{
    background: linear-gradient(135deg, #ebf8ff 0%, #e9d8fd 100%);
    border-left: 4px solid #667eea;
    border-radius: 6px;
    padding: 16px 20px;
    font-size: 0.9rem;
    line-height: 1.65;
    color: #2d3748;
    margin-bottom: 20px;
  }}
  .ai-summary .ai-label {{
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #667eea;
    margin-bottom: 8px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
  }}
  th {{
    background: #f7fafc;
    color: #718096;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: .05em;
    padding: 8px 10px;
    text-align: left;
    border-bottom: 2px solid #edf2f7;
  }}
  td {{
    padding: 8px 10px;
    border-bottom: 1px solid #f7fafc;
    vertical-align: middle;
  }}
  tr:hover td {{ background: #f7fafc; }}
  .issue-link {{ color: #4f46e5; font-weight: 600; text-decoration: none; }}
  .issue-link:hover {{ text-decoration: underline; }}
  .issue-summary {{ max-width: 340px; }}
  .sev-badge, .status-badge {{
    display: inline-block;
    color: white;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 999px;
    white-space: nowrap;
  }}
  .mini-bar-wrap {{
    background: #edf2f7;
    border-radius: 4px;
    height: 8px;
    width: 100px;
    overflow: hidden;
    display: inline-block;
    vertical-align: middle;
  }}
  .mini-bar {{
    height: 100%;
    border-radius: 4px;
  }}
  .sla-section-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 2px solid #edf2f7;
  }}
  .sla-section-title {{
    font-size: 0.95rem;
    font-weight: 700;
    color: #2d3748;
    flex: 1;
  }}
  .sla-policy-note {{
    font-size: 0.74rem;
    color: #718096;
    background: #f7fafc;
    border-radius: 6px;
    padding: 4px 10px;
  }}
  .sla-bar-wrap {{
    background: #edf2f7;
    border-radius: 6px;
    height: 10px;
    width: 120px;
    overflow: hidden;
    display: inline-block;
    vertical-align: middle;
  }}
  .sla-bar {{ height: 100%; border-radius: 6px; }}
  .sla-on-track  {{ background: #48bb78; }}
  .sla-at-risk   {{ background: #d69e2e; }}
  .sla-breached  {{ background: #e53e3e; }}
  .sla-not-started {{ background: #a0aec0; }}
  .sla-badge {{
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 999px;
    color: white;
    white-space: nowrap;
  }}
  .sla-badge.on-track   {{ background: #38a169; }}
  .sla-badge.at-risk    {{ background: #d69e2e; }}
  .sla-badge.breached   {{ background: #e53e3e; }}
  .sla-badge.not-started {{ background: #a0aec0; color: #2d3748; }}
  .kpi-card.sla-breached .kpi-value {{ color: #e53e3e; }}
  .kpi-card.sla-at-risk  .kpi-value {{ color: #d69e2e; }}
  .kpi-card.sla-ok       .kpi-value {{ color: #38a169; }}
</style>
</head>
<body data-fetched-at="{generated_at}">

<div class="page-header">
  <div>
    <h1>ðŸ› Production Bug Report â€” Jira</h1>
    <div class="subtitle">Project: PPPB Â· veradigm.atlassian.net</div>
  </div>
  <div class="header-meta">
    <div>Generated: {generated_at}</div>
    <div class="header-links">
      <a href="https://veradigm.atlassian.net/jira/software/c/projects/PPPB/boards/6973" target="_blank">ðŸ“‹ PPPB Board</a>
      <a href="https://veradigm.atlassian.net/issues/?jql=project%20%3D%20PPPB%20AND%20issuetype%20%3D%20Bug%20AND%20statusCategory%20!%3D%20Done%20ORDER%20BY%20priority%20DESC" target="_blank">ðŸ” All Open Prod Bugs</a>
    </div>
  </div>
</div>

<div class="migration-note">
  â„¹ï¸ <strong>Migration Note:</strong> Production bugs are now tracked in Jira (project PPPB) after migration from Azure DevOps (TFS).
  Historical ADO data is not fully imported, so trend data reflects only what is available in Jira.
  The legacy ADO Production Bug Report tab remains accessible for historical reference.
</div>

<!-- AI Summary -->
<div class="ai-summary">
  <div class="ai-label">ðŸ¤– AI Executive Summary</div>
  {ai_summary}
</div>

<!-- KPI Cards -->
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Total Open</div>
    <div class="kpi-value">{total}</div>
    <div class="kpi-delta">{d_total}</div>
  </div>
  <div class="kpi-card security">
    <div class="kpi-label">Security Bugs</div>
    <div class="kpi-value">{sec}</div>
    <div class="kpi-delta">{d_sec}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Non-Security</div>
    <div class="kpi-value">{non_sec}</div>
    <div class="kpi-delta"></div>
  </div>
  <div class="kpi-card critical">
    <div class="kpi-label">Critical</div>
    <div class="kpi-value">{crit}</div>
    <div class="kpi-delta">{d_crit}</div>
  </div>
  <div class="kpi-card high">
    <div class="kpi-label">High</div>
    <div class="kpi-value">{high}</div>
    <div class="kpi-delta">{d_high}</div>
  </div>
  <div class="kpi-card medium">
    <div class="kpi-label">Medium</div>
    <div class="kpi-value">{med}</div>
    <div class="kpi-delta"></div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Low</div>
    <div class="kpi-value">{low}</div>
    <div class="kpi-delta"></div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Aged &gt;90d</div>
    <div class="kpi-value" style="color:#dd6b20">{aging.get(">90d",0) + aging.get(">180d",0) + aging.get(">365d",0)}</div>
    <div class="kpi-delta"></div>
  </div>
</div>

<!-- SLA KPI Cards -->
<div class="section">
  <div class="sla-section-header">
    <div class="sla-section-title">â±ï¸ SLA Compliance â€” New Bugs (effective {metrics["sla_policy_start"]})</div>
    <div class="sla-policy-note">Critical: 3d &nbsp;Â·&nbsp; High: 14d &nbsp;Â·&nbsp; Medium: 21d &nbsp;Â·&nbsp; Low: 35d &nbsp;Â·&nbsp; Clock starts at In Progress</div>
  </div>
  <div class="kpi-grid" style="margin-bottom:0">
    <div class="kpi-card sla-ok">
      <div class="kpi-label">On Track</div>
      <div class="kpi-value">{metrics["sla_summary"].get("on_track", 0)}</div>
      <div class="kpi-delta" style="color:#38a169">â‰¤70% of SLA used</div>
    </div>
    <div class="kpi-card sla-at-risk">
      <div class="kpi-label">At Risk</div>
      <div class="kpi-value">{metrics["sla_summary"].get("at_risk", 0)}</div>
      <div class="kpi-delta" style="color:#d69e2e">70â€“100% of SLA used</div>
    </div>
    <div class="kpi-card sla-breached">
      <div class="kpi-label">Breached</div>
      <div class="kpi-value">{metrics["sla_summary"].get("breached", 0)}</div>
      <div class="kpi-delta" style="color:#e53e3e">SLA exceeded</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Not Started</div>
      <div class="kpi-value" style="color:#a0aec0">{metrics["sla_summary"].get("not_started", 0)}</div>
      <div class="kpi-delta">Awaiting triage / In Planning</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Backlog Exempt</div>
      <div class="kpi-value" style="color:#cbd5e0">{metrics["sla_summary"].get("exempt", 0)}</div>
      <div class="kpi-delta">Pre-policy bugs</div>
    </div>
  </div>
</div>

<!-- Charts -->
<div class="charts-grid">
  <div class="chart-section">
    {img(charts["severity_donut"], "Severity Donut")}
  </div>
  <div class="chart-section">
    {img(charts["security_bar"], "Security vs Non-Security")}
  </div>
  <div class="chart-section">
    {img(charts["aging"], "Aging Distribution")}
  </div>
  <div class="chart-section">
    {img(charts["trend"], "Monthly Trend")}
  </div>
  <div class="chart-section">
    {img(charts["sla_summary"], "SLA Compliance")}
  </div>
</div>

<!-- SLA Detail Table -->
<div class="section">
  <div class="sla-section-header">
    <div class="sla-section-title">â±ï¸ SLA Detail â€” New Bugs (breached & at-risk first)</div>
    <div class="sla-policy-note">Showing {len(metrics["sla_details"])} new bugs Â· {metrics["sla_summary"].get("exempt",0)} backlog bugs exempt</div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Key</th>
        <th>Severity</th>
        <th>Summary</th>
        <th>Status</th>
        <th>In Progress Since</th>
        <th>Elapsed / SLA</th>
        <th>Progress</th>
        <th>% Used</th>
        <th>SLA Status</th>
        <th>Assignee</th>
      </tr>
    </thead>
    <tbody>
      {sla_rows_html}
    </tbody>
  </table>
</div>

<!-- Top Open Bugs table -->
<div class="section">
  <div class="section-title">Top Open Production Bugs (by severity, then age)</div>
  <table>
    <thead>
      <tr>
        <th>Key</th>
        <th>Severity</th>
        <th>Summary</th>
        <th>Status</th>
        <th>Age</th>
        <th>Assignee</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</div>

<!-- Assignee table -->
<div class="section">
  <div class="section-title">Top Assignees â€” Open Bugs</div>
  <table>
    <thead><tr><th>Assignee</th><th>Distribution</th><th>Count</th></tr></thead>
    <tbody>{assignee_rows}</tbody>
  </table>
</div>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Confluence publisher
# ---------------------------------------------------------------------------

def _confluence_headers():
    return {"Accept": "application/json", "Content-Type": "application/json"}


def _get_confluence_space_id() -> str | None:
    """Return the first available Confluence space ID, preferring CONFLUENCE_SPACE_KEY."""
    url = f"{JIRA_BASE_URL}/wiki/api/v2/spaces"
    params = {"limit": 50}
    if CONFLUENCE_SPACE_KEY:
        params["keys"] = CONFLUENCE_SPACE_KEY
    r = requests.get(url, auth=AUTH, headers=_confluence_headers(), params=params, timeout=20)
    if not r.ok:
        print(f"  [Confluence] Could not list spaces: {r.status_code}")
        return None
    results = r.json().get("results", [])
    if not results:
        # Fall back to all spaces
        r2 = requests.get(url, auth=AUTH, headers=_confluence_headers(),
                          params={"limit": 10}, timeout=20)
        results = r2.json().get("results", []) if r2.ok else []
    if results:
        space = results[0]
        print(f"  [Confluence] Using space: {space.get('key')} â€” {space.get('name')}")
        return str(space["id"])
    return None


def _find_existing_page(space_id: str, title: str) -> dict | None:
    """Search for an existing page by title in a space."""
    r = requests.get(
        f"{JIRA_BASE_URL}/wiki/api/v2/pages",
        auth=AUTH, headers=_confluence_headers(),
        params={"spaceId": space_id, "title": title, "limit": 5},
        timeout=20,
    )
    if r.ok:
        results = r.json().get("results", [])
        if results:
            return results[0]
    return None


def _build_confluence_body(metrics: dict, ai_summary: str, charts: dict = None) -> str:
    """Build Confluence storage-format using HTML macro for full styled layout."""
    sev   = metrics["severity_counts"]
    sla   = metrics["sla_summary"]
    aging = metrics["aging_buckets"]
    total = metrics["total_open"]
    sec   = metrics["security_open"]
    non_sec = metrics["non_security_open"]
    gen   = metrics["generated_at"]

    crit  = sev.get("Critical", 0)
    high  = sev.get("High", 0)
    med   = sev.get("Medium", 0)
    low   = sev.get("Low", 0)
    aged_over_90 = aging.get(">90d", 0) + aging.get(">180d", 0) + aging.get(">365d", 0)

    board_url  = "https://veradigm.atlassian.net/jira/software/c/projects/PPPB/boards/6973"
    filter_url = "https://veradigm.atlassian.net/issues/?jql=project%20%3D%20PPPB%20AND%20issuetype%20%3D%20Bug%20AND%20statusCategory%20!%3D%20Done"

    # Chart images
    charts = charts or {}
    def img(key, alt=""):
        b64 = charts.get(key, "")
        if not b64:
            return ""
        return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;height:auto;border-radius:6px"/>'

    # Top issues table rows
    top_rows = ""
    for bug in metrics.get("top_issues", [])[:15]:
        f   = bug["fields"]
        key = bug["key"]
        url = f"{JIRA_BASE_URL}/browse/{key}"
        sev_val  = get_severity(bug)
        status   = (f.get("status") or {}).get("name", "Open")
        age      = age_days(f.get("created", ""))
        summary  = (f.get("summary") or "")[:80]
        assignee_i = f.get("assignee")
        assignee = (assignee_i or {}).get("displayName", "Unassigned") if assignee_i else "Unassigned"
        sec_flag = " ðŸ”’" if is_security(bug) else ""
        sev_colors = {"Critical": "#e53e3e", "High": "#dd6b20", "Medium": "#d69e2e", "Low": "#38a169"}
        badge_color = sev_colors.get(sev_val, "#a0aec0")
        top_rows += f"""<tr>
          <td><a href="{url}" style="color:#4f46e5;font-weight:600;text-decoration:none">{key}</a></td>
          <td><span style="background:{badge_color};color:white;font-size:11px;font-weight:700;padding:2px 8px;border-radius:999px">{sev_val}</span></td>
          <td style="max-width:300px">{summary}{sec_flag}</td>
          <td>{status}</td><td>{age}d</td><td>{assignee}</td>
        </tr>"""

    # ---- Native Confluence storage format (HTML macro is disabled on this instance) ----

    # Charts as Confluence attachments embedded inline
    chart_imgs = ""
    for key, alt in [("severity_donut","Severity"), ("security_bar","Security"),
                     ("aging","Aging Distribution"), ("trend","Monthly Trend"), ("sla_summary","SLA Compliance")]:
        b64 = charts.get(key, "")
        if b64:
            chart_imgs += f'<ac:image><ri:attachment ri:filename="chart_{key}.png"/></ac:image> '

    sev_colors = {"Critical": "Red", "High": "Yellow", "Medium": "Yellow", "Low": "Green"}

    body = f"""<p><em>Auto-generated: {gen} &nbsp;Â·&nbsp;
<a href="{board_url}">ðŸ“‹ PPPB Board</a> &nbsp;Â·&nbsp;
<a href="{filter_url}">ðŸ” All Open Prod Bugs</a></em></p>

<ac:structured-macro ac:name="info" ac:schema-version="1">
  <ac:parameter ac:name="title">Migration Note</ac:parameter>
  <ac:rich-text-body><p>Production bugs are now tracked in Jira project PPPB after migration from Azure DevOps (TFS). Historical ADO data is not fully imported.</p></ac:rich-text-body>
</ac:structured-macro>

<ac:structured-macro ac:name="panel" ac:schema-version="1">
  <ac:parameter ac:name="borderColor">#667eea</ac:parameter>
  <ac:parameter ac:name="titleBGColor">#ebf8ff</ac:parameter>
  <ac:parameter ac:name="borderStyle">solid</ac:parameter>
  <ac:parameter ac:name="title">ðŸ¤– AI Executive Summary</ac:parameter>
  <ac:rich-text-body><p>{ai_summary}</p></ac:rich-text-body>
</ac:structured-macro>

<h2>ðŸ“Š KPI Summary</h2>
<table>
  <tbody>
    <tr>
      <th>Total Open</th><th>Security</th><th>Non-Security</th>
      <th>Critical</th><th>High</th><th>Medium</th><th>Low</th><th>Aged &gt;90d</th>
    </tr>
    <tr>
      <td><strong><ac:structured-macro ac:name="status" ac:schema-version="1"><ac:parameter ac:name="colour">Blue</ac:parameter><ac:parameter ac:name="title">{total}</ac:parameter></ac:structured-macro></strong></td>
      <td><ac:structured-macro ac:name="status" ac:schema-version="1"><ac:parameter ac:name="colour">Red</ac:parameter><ac:parameter ac:name="title">{sec}</ac:parameter></ac:structured-macro></td>
      <td>{non_sec}</td>
      <td><ac:structured-macro ac:name="status" ac:schema-version="1"><ac:parameter ac:name="colour">Red</ac:parameter><ac:parameter ac:name="title">{crit}</ac:parameter></ac:structured-macro></td>
      <td><ac:structured-macro ac:name="status" ac:schema-version="1"><ac:parameter ac:name="colour">Yellow</ac:parameter><ac:parameter ac:name="title">{high}</ac:parameter></ac:structured-macro></td>
      <td><ac:structured-macro ac:name="status" ac:schema-version="1"><ac:parameter ac:name="colour">Yellow</ac:parameter><ac:parameter ac:name="title">{med}</ac:parameter></ac:structured-macro></td>
      <td><ac:structured-macro ac:name="status" ac:schema-version="1"><ac:parameter ac:name="colour">Green</ac:parameter><ac:parameter ac:name="title">{low}</ac:parameter></ac:structured-macro></td>
      <td>{aged_over_90}</td>
    </tr>
  </tbody>
</table>

<h2>â±ï¸ SLA Compliance â€” New Bugs (effective {metrics["sla_policy_start"]})</h2>
<p><em>Critical: 3d Â· High: 14d Â· Medium: 21d Â· Low: 35d Â· Clock starts at In Progress</em></p>
<table>
  <tbody>
    <tr><th>On Track</th><th>At Risk</th><th>Breached</th><th>Not Started</th><th>Backlog Exempt</th></tr>
    <tr>
      <td><ac:structured-macro ac:name="status" ac:schema-version="1"><ac:parameter ac:name="colour">Green</ac:parameter><ac:parameter ac:name="title">{sla.get("on_track",0)}</ac:parameter></ac:structured-macro></td>
      <td><ac:structured-macro ac:name="status" ac:schema-version="1"><ac:parameter ac:name="colour">Yellow</ac:parameter><ac:parameter ac:name="title">{sla.get("at_risk",0)}</ac:parameter></ac:structured-macro></td>
      <td><ac:structured-macro ac:name="status" ac:schema-version="1"><ac:parameter ac:name="colour">Red</ac:parameter><ac:parameter ac:name="title">{sla.get("breached",0)}</ac:parameter></ac:structured-macro></td>
      <td>{sla.get("not_started",0)}</td>
      <td>{sla.get("exempt",0)}</td>
    </tr>
  </tbody>
</table>

<h2>ðŸ“ˆ Charts</h2>
<ac:layout>
  <ac:layout-section ac:type="three_equal">
    <ac:layout-cell>
      <p><strong>Open Bugs by Severity</strong></p>
      <ac:image ac:width="300"><ri:attachment ri:filename="chart_severity_donut.png"/></ac:image>
    </ac:layout-cell>
    <ac:layout-cell>
      <p><strong>Security vs Non-Security</strong></p>
      <ac:image ac:width="300"><ri:attachment ri:filename="chart_security_bar.png"/></ac:image>
    </ac:layout-cell>
    <ac:layout-cell>
      <p><strong>Aging Distribution</strong></p>
      <ac:image ac:width="300"><ri:attachment ri:filename="chart_aging.png"/></ac:image>
    </ac:layout-cell>
  </ac:layout-section>
  <ac:layout-section ac:type="two_equal">
    <ac:layout-cell>
      <p><strong>Monthly Bug Opened vs Closed</strong></p>
      <ac:image ac:width="460"><ri:attachment ri:filename="chart_trend.png"/></ac:image>
    </ac:layout-cell>
    <ac:layout-cell>
      <p><strong>SLA Compliance</strong></p>
      <ac:image ac:width="460"><ri:attachment ri:filename="chart_sla_summary.png"/></ac:image>
    </ac:layout-cell>
  </ac:layout-section>
</ac:layout>

<h2>ðŸ”´ Top Open Production Bugs</h2>
<table>
  <tbody>
    <tr><th>Key</th><th>Severity</th><th>Summary</th><th>Status</th><th>Age</th><th>Assignee</th></tr>
    {top_rows}
  </tbody>
</table>
"""
    return body


def publish_to_confluence(metrics: dict, ai_summary: str, charts: dict = None):
    """Create or update the Confluence report page."""
    print("Publishing to Confluenceâ€¦")

    def _set_full_width(page_id):
        for key in ["content-appearance-published", "content-appearance-draft"]:
            url = f"{JIRA_BASE_URL}/wiki/rest/api/content/{page_id}/property/{key}"
            r = requests.get(url, auth=AUTH, headers=_confluence_headers(), timeout=10)
            if r.ok:
                ver = r.json().get("version", {}).get("number", 1)
                requests.put(url, auth=AUTH, headers=_confluence_headers(),
                             json={"key": key, "value": "full-width", "version": {"number": ver + 1}},
                             timeout=10)
            else:
                requests.post(f"{JIRA_BASE_URL}/wiki/rest/api/content/{page_id}/property",
                              auth=AUTH, headers=_confluence_headers(),
                              json={"key": key, "value": "full-width"}, timeout=10)

    def _upload_charts(page_id):
        if not charts:
            return
        chart_names = {
            "severity_donut": "chart_severity_donut.png",
            "security_bar":   "chart_security_bar.png",
            "aging":          "chart_aging.png",
            "trend":          "chart_trend.png",
            "sla_summary":    "chart_sla_summary.png",
        }
        # Fetch existing attachment IDs
        existing = {}
        r0 = requests.get(
            f"{JIRA_BASE_URL}/wiki/rest/api/content/{page_id}/child/attachment",
            auth=AUTH, headers={"Accept": "application/json"},
            params={"limit": 50}, timeout=20,
        )
        if r0.ok:
            for att in r0.json().get("results", []):
                existing[att["title"]] = att["id"]

        for key, filename in chart_names.items():
            b64 = charts.get(key, "")
            if not b64:
                continue
            img_bytes = base64.b64decode(b64)
            att_id = existing.get(filename)
            if att_id:
                url = f"{JIRA_BASE_URL}/wiki/rest/api/content/{page_id}/child/attachment/{att_id}/data"
            else:
                url = f"{JIRA_BASE_URL}/wiki/rest/api/content/{page_id}/child/attachment"
            r = requests.post(
                url, auth=AUTH,
                headers={"X-Atlassian-Token": "no-check"},
                files={"file": (filename, img_bytes, "image/png")},
                timeout=30,
            )
            if r.ok:
                print(f"    Uploaded {filename}")
            else:
                print(f"    Chart upload failed {filename}: {r.status_code} {r.text[:100]}")

    # Try to reuse a cached page ID first
    cached_id = None
    if os.path.isfile(CONFLUENCE_PAGE_ID_CACHE):
        try:
            cached_id = open(CONFLUENCE_PAGE_ID_CACHE).read().strip()
        except Exception:
            pass

    body_content = _build_confluence_body(metrics, ai_summary, charts)

    if cached_id:
        # Fetch current version
        r = requests.get(f"{JIRA_BASE_URL}/wiki/api/v2/pages/{cached_id}",
                         auth=AUTH, headers=_confluence_headers(), timeout=20)
        if r.ok:
            page = r.json()
            current_version = page.get("version", {}).get("number", 1)
            payload = {
                "id": cached_id,
                "status": "current",
                "title": CONFLUENCE_PAGE_TITLE,
                "body": {"representation": "storage", "value": body_content},
                "version": {"number": current_version + 1},
            }
            r2 = requests.put(
                f"{JIRA_BASE_URL}/wiki/api/v2/pages/{cached_id}",
                auth=AUTH, headers=_confluence_headers(), json=payload, timeout=30,
            )
            if r2.ok:
                links = r2.json().get("_links", {})
                page_url = JIRA_BASE_URL + "/wiki" + links.get("webui", f"/pages/{cached_id}")
                print(f"  [Confluence] Updated page: {page_url}")
                _set_full_width(cached_id)
                _upload_charts(cached_id)
                return
            print(f"  [Confluence] Update failed ({r2.status_code}), will recreateâ€¦")

    # Create new page
    space_id = _get_confluence_space_id()
    if not space_id:
        print("  [Confluence] No space found â€” skipping publish.")
        return

    # Check if page already exists
    existing = _find_existing_page(space_id, CONFLUENCE_PAGE_TITLE)
    if existing:
        cached_id = str(existing["id"])
        current_version = existing.get("version", {}).get("number", 1)
        payload = {
            "id": cached_id,
            "status": "current",
            "title": CONFLUENCE_PAGE_TITLE,
            "body": {"representation": "storage", "value": body_content},
            "version": {"number": current_version + 1},
        }
        r = requests.put(
            f"{JIRA_BASE_URL}/wiki/api/v2/pages/{cached_id}",
            auth=AUTH, headers=_confluence_headers(), json=payload, timeout=30,
        )
        if r.ok:
            print(f"  [Confluence] Updated existing page id={cached_id}")
            open(CONFLUENCE_PAGE_ID_CACHE, "w").write(cached_id)
            _set_full_width(cached_id)
            _upload_charts(cached_id)
        else:
            print(f"  [Confluence] Update failed: {r.status_code} {r.text[:200]}")
        return

    # Fresh create
    payload = {
        "spaceId": space_id,
        "status": "current",
        "title": CONFLUENCE_PAGE_TITLE,
        "body": {"representation": "storage", "value": body_content},
    }
    r = requests.post(
        f"{JIRA_BASE_URL}/wiki/api/v2/pages",
        auth=AUTH, headers=_confluence_headers(), json=payload, timeout=30,
    )
    if r.ok:
        page_id = str(r.json()["id"])
        open(CONFLUENCE_PAGE_ID_CACHE, "w").write(page_id)
        links = r.json().get("_links", {})
        page_url = JIRA_BASE_URL + "/wiki" + links.get("webui", f"/pages/{page_id}")
        print(f"  [Confluence] Created page id={page_id} â€” {page_url}")
        _set_full_width(page_id)
        _upload_charts(page_id)
    else:
        print(f"  [Confluence] Create failed: {r.status_code} {r.text[:300]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Fetching open production bugs from Jira (PPPB)â€¦")
    jql_open = (
        f'project = {JIRA_PROJECT} AND issuetype = Bug '
        f'AND statusCategory != Done '
        f'ORDER BY priority ASC, created ASC'
    )
    open_bugs = fetch_all_bugs(jql_open)
    print(f"  Found {len(open_bugs)} open bugs")

    print("Fetching recently closed/opened bugs for trendâ€¦")
    closed_bugs = fetch_recent_closed_bugs(days=180)
    opened_bugs = fetch_recently_opened_bugs(days=180)
    print(f"  Closed (last 180d): {len(closed_bugs)}  |  Opened (last 180d): {len(opened_bugs)}")

    print("Fetching SLA changelog dataâ€¦")
    sla_start_dates = fetch_sla_start_dates(open_bugs)

    metrics = compute_metrics(open_bugs, closed_bugs, opened_bugs, sla_start_dates)
    prev    = load_previous_counts()

    print("Generating chartsâ€¦")
    charts = {
        "severity_donut": chart_severity_donut(metrics["severity_counts"]),
        "security_bar":   chart_security_bar(metrics["security_open"], metrics["non_security_open"]),
        "aging":          chart_aging(metrics["aging_buckets"]),
        "trend":          chart_trend(metrics["trend"]),
        "sla_summary":    chart_sla_summary(metrics["sla_summary"]),
    }

    print("Generating AI executive summaryâ€¦")
    ai_summary = generate_ai_summary(metrics)

    print("Rendering HTMLâ€¦")
    html = render_html(metrics, charts, ai_summary, prev)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved to: {OUTPUT_PATH}")

    save_current_counts(metrics)

    publish_to_confluence(metrics, ai_summary, charts)
    print("Done.")


if __name__ == "__main__":
    main()
