from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialize once (avoid re-creating client for every call)
MEGALLM_BASE_URL = os.getenv("MEGALLM_BASE_URL", "https://ai.megallm.io/v1")
MEGALLM_API_KEY = os.getenv("MEGALLM_API_KEY", "")  # set via env for safety
MEGALLM_MODEL = os.getenv("MEGALLM_MODEL", "deepseek-r1-distill-llama-70b")

client = OpenAI(
    base_url=MEGALLM_BASE_URL,
    api_key=MEGALLM_API_KEY,
)


def llm_generate(prompt: str) -> str:
    """
    Contract:
      - Input : prompt (str)
      - Output: text  (str)
    """
    # You can tune these
    max_retries = 6
    temperature = 0.2
    max_tokens = 256

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MEGALLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful finance data generator. "
                            "Follow the user's instructions exactly. "
                            "Return ONLY the requested text with no extra commentary."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            text = (resp.choices[0].message.content or "").strip()
            return text

        except Exception as e:
            # Log detailed error information
            logging.error(
                f"LLM API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}"
            )
            # basic exponential backoff
            sleep_s = min(60.0, (2**attempt) + random.random())
            logging.warning(f"Retrying in {sleep_s:.1f} seconds...")
            time.sleep(sleep_s)

    logging.error("LLM generation failed after all retries")
    return ""  # fallback if all retries fail


def llm_paraphrase_evidence(evidence: str) -> str:
    """
    Use LLM to paraphrase evidence while preserving meaning.
    Returns original if LLM fails.
    """
    prompt = (
        "Rewrite the following evidence paragraph in natural, plain English. "
        "Keep the same meaning and all key facts. Do not add any new information.\n\n"
        f"{evidence}"
    )
    result = llm_generate(prompt)
    return _norm_ws(result) if result else evidence


def llm_paraphrase_claim(claim: str) -> str:
    """
    Use LLM to paraphrase claim while preserving meaning.
    Returns original if LLM fails.
    """
    prompt = (
        "Rewrite the following claim as a concise, news-like sentence. "
        "Keep the exact same meaning. Do not change any facts.\n\n"
        f"{claim}"
    )
    result = llm_generate(prompt)
    return _norm_ws(result) if result else claim


def make_hard_contradiction(
    evidence: str, base_concept: str, rng: random.Random
) -> str:
    """
    Create a claim that clearly contradicts the evidence.
    Uses LLM with careful prompting to generate controlled contradictions.
    """
    prompt = (
        "Based on this evidence, generate a SHORT claim (1 sentence) that directly CONTRADICTS it. "
        "The claim should be clearly false based on the evidence.\n\n"
        f"Evidence: {evidence}\n\n"
        f"Topic hint: {base_concept}\n\n"
        "Generate ONLY the contradictory claim, nothing else:"
    )
    result = llm_generate(prompt)
    return (
        _norm_ws(result)
        if result
        else f"The opposite of the evidence is true regarding {base_concept}."
    )


def make_hard_unsupported(evidence: str, base_concept: str, rng: random.Random) -> str:
    """
    Create a claim that adds unsupported absolute conditions or specific numbers.
    The claim is plausible but evidence doesn't support the absolute/specific assertion.
    """
    absolute_phrases = [
        "in every single case",
        "with 100% certainty",
        "always without exception",
        "as officially confirmed by all regulators",
        "guarantees exactly 50% returns",
        "is required by law in all countries",
        "never varies under any circumstances",
    ]

    modifier = rng.choice(absolute_phrases)

    prompt = (
        "Based on this evidence, generate a SHORT claim (1 sentence) that is related to the topic "
        f"and includes this absolute assertion: '{modifier}'. "
        "The claim should sound plausible but the evidence does NOT support the absolute assertion.\n\n"
        f"Evidence: {evidence}\n\n"
        "Generate ONLY the claim, nothing else:"
    )
    result = llm_generate(prompt)
    return (
        _norm_ws(result)
        if result
        else f"This concept {modifier} regarding {base_concept}."
    )


# =============================
# 2) Concept bank
# =============================


@dataclass(frozen=True)
class Concept:
    topic: str
    true_stmt: str  # supported by evidence
    evidence_core: str
    contradict_stmt: str  # contradicts evidence


CONCEPTS: List[Concept] = [
    Concept(
        topic="bonds_rates",
        true_stmt="Bond prices typically move inversely to interest rates.",
        evidence_core="When market interest rates rise, newly issued bonds tend to offer higher yields. Existing lower-yield bonds become less attractive, so their prices generally fall; when rates fall, bond prices often rise.",
        contradict_stmt="Bond prices typically rise when interest rates rise.",
    ),
    Concept(
        topic="inflation_pp",
        true_stmt="Higher inflation tends to reduce the purchasing power of money over time.",
        evidence_core="Inflation is a broad rise in prices. If prices increase while income does not rise proportionally, the same amount of money buys fewer goods and services, reducing purchasing power.",
        contradict_stmt="Higher inflation tends to increase the purchasing power of money over time.",
    ),
    Concept(
        topic="real_vs_nominal",
        true_stmt="Real returns adjust nominal returns for inflation.",
        evidence_core="A nominal return is the stated percentage gain without accounting for inflation. A real return reflects purchasing power by adjusting nominal returns for inflation, often approximated as nominal return minus inflation.",
        contradict_stmt="Real returns ignore inflation, while nominal returns account for inflation.",
    ),
    Concept(
        topic="compounding",
        true_stmt="Compounding allows interest to be earned on previously earned interest.",
        evidence_core="With compounding, interest is added to the principal, so future interest is calculated on a larger base. Over time, this can cause balances to grow faster than with simple interest.",
        contradict_stmt="Compounding prevents interest from being earned on previously earned interest.",
    ),
    Concept(
        topic="present_value",
        true_stmt="Present value discounts future cash flows to reflect the time value of money.",
        evidence_core="Money today can be worth more than the same amount in the future because it can earn returns. Present value converts future cash flows into an equivalent value today using a discount rate.",
        contradict_stmt="Present value increases future cash flows so they are larger than today.",
    ),
    Concept(
        topic="npv",
        true_stmt="A positive net present value (NPV) indicates expected value creation under the chosen assumptions.",
        evidence_core="NPV compares the present value of expected future cash inflows to the present value of costs. If discounted inflows exceed costs, the project is expected to add value given the assumptions and discount rate.",
        contradict_stmt="A positive NPV indicates expected value destruction under the chosen assumptions.",
    ),
    Concept(
        topic="wacc",
        true_stmt="Weighted average cost of capital (WACC) reflects a firm's average cost of financing from debt and equity.",
        evidence_core="WACC combines the cost of equity and the after-tax cost of debt in proportion to how much the firm uses each source. It is often used as a discount rate for cash flows that reflect the overall firm.",
        contradict_stmt="WACC reflects only a firm's dividend payout to shareholders.",
    ),
    Concept(
        topic="diversification",
        true_stmt="Diversification can reduce portfolio risk by spreading exposure across different assets.",
        evidence_core="If assets do not move perfectly together, losses in one position may be offset by gains or smaller losses in another. This can reduce overall portfolio volatility relative to a concentrated position.",
        contradict_stmt="Diversification always increases portfolio risk by spreading exposure.",
    ),
    Concept(
        topic="idiosyncratic_vs_systematic",
        true_stmt="Diversification tends to reduce idiosyncratic risk more than systematic market risk.",
        evidence_core="Idiosyncratic risk is specific to a company or asset and can be reduced by holding many different assets. Systematic risk affects broad markets and cannot be eliminated simply by diversifying within the same market.",
        contradict_stmt="Diversification eliminates all market-wide risk.",
    ),
    Concept(
        topic="risk_return",
        true_stmt="Higher expected returns are often associated with higher risk, but higher risk does not guarantee higher realized returns.",
        evidence_core="Riskier assets may offer higher expected compensation to investors. However, outcomes vary and taking more risk does not ensure higher realized returns in any particular period.",
        contradict_stmt="Higher expected returns are always associated with lower risk.",
    ),
    Concept(
        topic="beta",
        true_stmt="Beta is commonly used to describe how sensitive an asset is to broad market movements.",
        evidence_core="Beta is often interpreted as how much an asset's returns tend to move relative to a market benchmark. A beta above 1 suggests greater sensitivity than the market, while below 1 suggests less.",
        contradict_stmt="Beta measures a company's revenue in dollars.",
    ),
    Concept(
        topic="volatility",
        true_stmt="Volatility measures how much an asset's price or returns fluctuate over time.",
        evidence_core="Volatility is commonly measured using the standard deviation of returns. Higher volatility means larger swings in price and is often interpreted as greater uncertainty.",
        contradict_stmt="Volatility means an asset's price never changes.",
    ),
    Concept(
        topic="bid_ask",
        true_stmt="The bid-ask spread is the difference between the best available bid and the best available ask.",
        evidence_core="In many markets, the bid is the highest price a buyer is willing to pay and the ask is the lowest price a seller is willing to accept. The spread reflects liquidity, trading costs, and market-making compensation.",
        contradict_stmt="The bid-ask spread is the total number of shares outstanding.",
    ),
    Concept(
        topic="market_order",
        true_stmt="A market order prioritizes immediate execution at the best available price.",
        evidence_core="Market orders emphasize execution speed over price certainty. The execution price depends on the order book and available liquidity at the moment the order is filled.",
        contradict_stmt="A market order guarantees execution at a pre-set specific price.",
    ),
    Concept(
        topic="limit_order",
        true_stmt="A limit order sets a maximum buy price or a minimum sell price.",
        evidence_core="Limit orders provide price control but may not execute if the market does not reach the limit. A buy limit executes at the limit price or lower, and a sell limit executes at the limit price or higher.",
        contradict_stmt="A limit order always executes immediately regardless of the market price.",
    ),
    Concept(
        topic="liquidity",
        true_stmt="Liquidity reflects how easily an asset can be traded without materially affecting its price.",
        evidence_core="Highly liquid markets tend to have many participants and narrow bid-ask spreads, allowing trades with relatively small price impact. Illiquid assets may require accepting a worse price to trade quickly.",
        contradict_stmt="Liquidity means it is difficult to trade an asset quickly.",
    ),
    Concept(
        topic="market_cap",
        true_stmt="Market capitalization is computed as share price times shares outstanding.",
        evidence_core="Market cap is a market-based measure of a company's equity value. Because share prices move over time, market cap can also change even if shares outstanding remain constant.",
        contradict_stmt="Market capitalization is computed as revenue divided by expenses.",
    ),
    Concept(
        topic="enterprise_value",
        true_stmt="Enterprise value is a firm value measure that commonly adjusts equity value for debt and cash.",
        evidence_core="Enterprise value is often used to approximate the value of the operating business by combining equity value with net debt. Specific formulas vary, but the concept is to reflect total value beyond just equity.",
        contradict_stmt="Enterprise value is identical to a company's revenue.",
    ),
    Concept(
        topic="pe_ratio",
        true_stmt="A price-to-earnings ratio compares a share price to earnings per share.",
        evidence_core="The P/E ratio is commonly computed as price per share divided by earnings per share. It is widely used as a valuation multiple, but interpretation depends on growth, risk, and accounting choices.",
        contradict_stmt="A P/E ratio compares a company's debt to its cash flow.",
    ),
    Concept(
        topic="eps",
        true_stmt="Earnings per share (EPS) is net income divided by the weighted average number of shares.",
        evidence_core="EPS expresses profit on a per-share basis. Because share counts can change due to issuance or buybacks, EPS often uses a weighted average share count over the reporting period.",
        contradict_stmt="EPS is total assets divided by total liabilities.",
    ),
    Concept(
        topic="dividends",
        true_stmt="Dividends are discretionary distributions a company may pay to shareholders.",
        evidence_core="Companies may choose to pay dividends as a way to return value to shareholders, often in cash or additional shares. Dividend policies vary and many companies choose not to pay dividends.",
        contradict_stmt="Dividends are mandatory fines that shareholders must pay to the company.",
    ),
    Concept(
        topic="buybacks",
        true_stmt="Share buybacks reduce shares outstanding and can raise EPS if earnings stay constant.",
        evidence_core="When a company repurchases its shares, the share count decreases. If net income is unchanged, dividing by fewer shares results in a higher EPS, though total value depends on buyback price and other factors.",
        contradict_stmt="Share buybacks increase shares outstanding.",
    ),
    Concept(
        topic="stock_split",
        true_stmt="A stock split increases the number of shares while reducing the price per share proportionally.",
        evidence_core="In a stock split, each existing share is divided into multiple shares. The price per share adjusts so that the total equity value remains the same immediately after the split, absent market reactions.",
        contradict_stmt="A stock split directly creates new company profits.",
    ),
    Concept(
        topic="dilution",
        true_stmt="Issuing new shares can dilute existing shareholders' ownership percentage.",
        evidence_core="When additional shares are issued, the total number of shares increases. If an existing shareholder does not buy additional shares, their ownership percentage can decline.",
        contradict_stmt="Issuing new shares always increases each shareholder's ownership percentage.",
    ),
    Concept(
        topic="current_ratio",
        true_stmt="The current ratio equals current assets divided by current liabilities.",
        evidence_core="The current ratio is a liquidity metric comparing resources expected to be converted to cash within a year against obligations due within a year. Interpretation depends on industry and asset quality.",
        contradict_stmt="The current ratio equals total revenue divided by total expenses.",
    ),
    Concept(
        topic="working_capital",
        true_stmt="Working capital is commonly defined as current assets minus current liabilities.",
        evidence_core="Working capital reflects short-term financial flexibility. Positive working capital can indicate capacity to cover near-term obligations, while negative working capital may indicate tighter liquidity depending on the business model.",
        contradict_stmt="Working capital is total liabilities minus total assets.",
    ),
    Concept(
        topic="accrual_vs_cash",
        true_stmt="Accrual accounting recognizes revenue when earned rather than when cash is received.",
        evidence_core="Under accrual accounting, revenue and expenses are recorded when they are earned or incurred. This can cause accounting profit to differ from cash flow due to timing differences in collections and payments.",
        contradict_stmt="Accrual accounting records revenue only when cash is received.",
    ),
    Concept(
        topic="profit_vs_cashflow",
        true_stmt="A company can report accounting profits while having negative operating cash flow in a period.",
        evidence_core="Accrual accounting recognizes revenue and expenses when earned or incurred, not necessarily when cash changes hands. Working capital changes can make operating cash flow diverge from net income.",
        contradict_stmt="If a company reports profit, operating cash flow must be positive in the same period.",
    ),
    Concept(
        topic="depreciation",
        true_stmt="Depreciation allocates the cost of a long-lived asset over its useful life and is typically a non-cash expense.",
        evidence_core="Depreciation reduces reported accounting profit but does not represent an immediate cash outflow in the period it is recorded. Cash impact typically occurred when the asset was purchased.",
        contradict_stmt="Depreciation is a cash payment made to suppliers every month.",
    ),
    Concept(
        topic="amortization",
        true_stmt="Amortization spreads the cost of certain intangible assets over time.",
        evidence_core="Amortization is an accounting process that allocates the cost of an intangible asset across its useful life. Like depreciation, it is generally a non-cash expense in the period recognized.",
        contradict_stmt="Amortization is the interest rate on a credit card.",
    ),
    Concept(
        topic="ebitda",
        true_stmt="EBITDA is earnings before interest, taxes, depreciation, and amortization.",
        evidence_core="EBITDA is a non-GAAP metric often used to approximate operating performance by excluding financing costs, taxes, and certain non-cash charges. It is not the same as cash flow and can omit important costs.",
        contradict_stmt="EBITDA is always identical to net cash flow.",
    ),
    Concept(
        topic="leverage",
        true_stmt="Financial leverage can amplify both gains and losses for equity holders.",
        evidence_core="Debt financing introduces fixed obligations such as interest payments. When performance is strong, leverage can increase returns on equity, but when performance weakens, leverage can magnify losses and increase default risk.",
        contradict_stmt="Financial leverage always reduces risk for equity holders.",
    ),
    Concept(
        topic="credit_risk",
        true_stmt="Credit risk is the risk that a borrower will fail to make required payments.",
        evidence_core="In lending and bond markets, credit risk refers to the possibility of default or missed payments, which can lead to losses for lenders or bondholders.",
        contradict_stmt="Credit risk is the risk that a currency symbol changes.",
    ),
    Concept(
        topic="credit_spread",
        true_stmt="A credit spread is the yield difference between a riskier bond and a safer benchmark bond.",
        evidence_core="Investors may demand extra yield to compensate for default risk and other risks. The difference in yields between similar-maturity bonds with different credit quality is commonly called a credit spread.",
        contradict_stmt="A credit spread is the same thing as a company's dividend policy.",
    ),
    Concept(
        topic="ytm",
        true_stmt="Yield to maturity estimates the annualized return of a bond if held to maturity under standard assumptions.",
        evidence_core="Yield to maturity is the rate that discounts a bond's expected cash flows, such as coupons and principal repayment, to its current market price. It typically assumes payments occur as scheduled.",
        contradict_stmt="Yield to maturity is the same as a bond's face value.",
    ),
    Concept(
        topic="duration",
        true_stmt="Duration is a measure of a bond's price sensitivity to changes in yields.",
        evidence_core="Duration summarizes how much a bond's price tends to change for a given change in yields. Longer duration generally implies greater price sensitivity to interest rate movements.",
        contradict_stmt="Duration is the bond's coupon rate.",
    ),
    Concept(
        topic="yield_curve",
        true_stmt="A yield curve plots bond yields against different maturities.",
        evidence_core="A yield curve shows interest rates or yields for bonds of different maturities, often government bonds. The curve's shape can change over time and is sometimes used to summarize expectations about growth, inflation, and policy.",
        contradict_stmt="A yield curve plots a company's quarterly sales.",
    ),
    Concept(
        topic="fx_rate",
        true_stmt="An exchange rate indicates how much of one currency is needed to buy one unit of another currency.",
        evidence_core="Exchange rates enable conversion between currencies. They can move due to many factors such as interest rate differences, inflation expectations, risk sentiment, and capital flows.",
        contradict_stmt="An exchange rate is the fee charged on a credit card payment.",
    ),
    Concept(
        topic="call_option",
        true_stmt="A call option gives the holder the right, but not the obligation, to buy an asset at a specified price.",
        evidence_core="Options are derivatives. A call option allows the buyer to purchase the underlying at the strike price by expiration; the holder can let it expire if exercising is unfavorable.",
        contradict_stmt="A call option obligates the holder to buy the asset regardless of price.",
    ),
    Concept(
        topic="put_option",
        true_stmt="A put option gives the holder the right, but not the obligation, to sell an asset at a specified price.",
        evidence_core="A put option allows the buyer to sell the underlying at the strike price by expiration. The holder may choose not to exercise if it is not beneficial.",
        contradict_stmt="A put option obligates the holder to buy the asset at the strike price.",
    ),
    Concept(
        topic="futures",
        true_stmt="A futures contract is an agreement to buy or sell an underlying asset at a predetermined price at a future date.",
        evidence_core="Futures are standardized contracts often traded on exchanges. They are used for hedging or speculation and typically involve margining that can settle gains and losses over time.",
        contradict_stmt="A futures contract is a bank deposit account.",
    ),
    Concept(
        topic="hedging",
        true_stmt="Hedging uses an offsetting position to reduce exposure to a particular risk.",
        evidence_core="A hedge is designed to reduce potential losses from an adverse price movement. Hedging can reduce upside potential and may involve costs such as premiums or transaction fees.",
        contradict_stmt="Hedging increases exposure to the same risk.",
    ),
    Concept(
        topic="short_selling",
        true_stmt="Short selling involves selling borrowed shares with the intention of buying them back later.",
        evidence_core="In a typical short sale, shares are borrowed and sold. The short seller profits if the price falls and they can repurchase at a lower price, but losses can grow if the price rises.",
        contradict_stmt="Short selling involves buying shares and holding them for decades.",
    ),
    Concept(
        topic="etf",
        true_stmt="An exchange-traded fund (ETF) holds a basket of assets and trades on an exchange like a stock.",
        evidence_core="ETFs often track an index or strategy and can be bought or sold throughout the trading day. Their market price may differ from net asset value depending on liquidity and market conditions.",
        contradict_stmt="An ETF is a private loan that cannot be traded.",
    ),
    Concept(
        topic="mutual_fund_nav",
        true_stmt="Net asset value (NAV) reflects a fund's assets minus liabilities, typically expressed per share.",
        evidence_core="NAV is calculated by valuing holdings, subtracting liabilities, and dividing by shares outstanding. Depending on the fund structure, transactions may occur at or around NAV.",
        contradict_stmt="NAV is the number of employees working at the fund.",
    ),
    Concept(
        topic="central_bank_rates",
        true_stmt="Central banks may adjust policy rates to influence borrowing conditions and inflation.",
        evidence_core="Policy rates can affect interest rates throughout the economy. Raising rates can discourage borrowing and spending, while lowering rates can encourage them, with implications for inflation and growth.",
        contradict_stmt="Central banks change policy rates only to affect corporate branding.",
    ),
    Concept(
        topic="audit_assurance",
        true_stmt="External audits typically provide reasonable assurance rather than absolute certainty.",
        evidence_core="Audits use sampling, testing, and professional judgment to reduce the risk of material misstatement. Because of inherent limitations, audits generally provide reasonable assurance, not a guarantee of perfect accuracy.",
        contradict_stmt="An external audit guarantees that financial statements contain no errors.",
    ),
    Concept(
        topic="gaap_ifrs",
        true_stmt="GAAP and IFRS are accounting standards frameworks used to prepare financial statements.",
        evidence_core="Companies prepare financial statements following a set of accounting standards. GAAP and IFRS are two widely used frameworks, and specific rules can differ between them.",
        contradict_stmt="GAAP and IFRS are stock market indices.",
    ),
    Concept(
        topic="insider_trading",
        true_stmt="Insider trading generally refers to trading based on material non-public information.",
        evidence_core="Many jurisdictions restrict trading on material information that is not publicly available to reduce unfair informational advantages and protect market integrity.",
        contradict_stmt="Insider trading means trading only during lunch breaks.",
    ),
]


# =============================
# 3) Templates for extra variety
# =============================

TRUE_PREFIXES = [
    "",
    "In general, ",
    "Typically, ",
    "In many financial markets, ",
    "As a rule of thumb, ",
    "A common finance principle is that ",
]

FALSE_PREFIXES = [
    "",
    "It is correct that ",
    "It is a fact that ",
    "In all cases, ",
]

UNSUPPORTED_TAILS = [
    " in every situation.",
    " with no exceptions.",
    ", regardless of market conditions.",
    ", and this guarantees profits for investors.",
    ", as officially confirmed by regulators in every country.",
]

EVIDENCE_FILLERS = [
    "This is a general concept often taught in introductory finance.",
    "The exact magnitude of the effect can vary with market structure and assumptions.",
    "Real-world outcomes can differ depending on timing, liquidity, and investor behavior.",
    "Definitions may differ slightly across jurisdictions and reporting standards.",
    "These relationships are commonly discussed in basic financial analysis.",
    "In practice, multiple factors can influence the observed result.",
]


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _strip_period(s: str) -> str:
    s = _norm_ws(s)
    return s[:-1] if s.endswith(".") else s


def _lc_first(s: str) -> str:
    s = _norm_ws(s)
    return (s[0].lower() + s[1:]) if s else s


def make_true_claim(c: Concept, rng: random.Random) -> str:
    base = _strip_period(c.true_stmt)
    prefix = rng.choice(TRUE_PREFIXES)
    if prefix in {"A common finance principle is that ", ""}:
        stmt = _lc_first(base) if prefix.endswith("that ") else base
        out = prefix + stmt
    else:
        out = prefix + _lc_first(base) if prefix else base
    return _norm_ws(out) + "."


def make_false_contradict_claim(c: Concept, rng: random.Random) -> str:
    base = _strip_period(c.contradict_stmt)
    prefix = rng.choice(FALSE_PREFIXES)
    out = prefix + (_lc_first(base) if prefix else base)
    return _norm_ws(out) + "."


def make_false_unsupported_claim(c: Concept, rng: random.Random) -> str:
    # Unsupported = plausible but evidence does not confirm (extra absolute / extra assertion)
    base = _strip_period(c.true_stmt)
    prefix = rng.choice(FALSE_PREFIXES)
    tail = rng.choice(UNSUPPORTED_TAILS)
    out = prefix + (_lc_first(base) if prefix else base)
    # Ensure one sentence (avoid double period)
    out = _strip_period(_norm_ws(out)) + tail
    return _norm_ws(out)


def make_evidence(
    c: Concept, rng: random.Random, min_fillers: int, max_fillers: int
) -> str:
    core = _norm_ws(c.evidence_core)
    k = rng.randint(min_fillers, max_fillers)
    chosen = (
        rng.sample(EVIDENCE_FILLERS, k=min(k, len(EVIDENCE_FILLERS))) if k > 0 else []
    )
    evidence = core + (" " + " ".join(chosen) if chosen else "")
    return _norm_ws(evidence)


# =============================
# 4) Dataset builder
# =============================


def build_rows(seed: int) -> List[dict]:
    """
    Build 1500 samples with controlled quality:
    - 400 controlled samples (pure rule-based, no paraphrase)
    - 500 controlled samples (rule-based + LLM paraphrase for natural language)
    - 600 hard set samples (LLM-generated contradiction + unsupported)

    This ensures clean labels and tests fact-checking capability.
    """
    rng = random.Random(seed)

    # Fixed parameters for evidence generation
    min_ev_fillers = 2
    max_ev_fillers = 5

    # Cycle concepts to avoid over-using only a few
    order = list(range(len(CONCEPTS)))
    rng.shuffle(order)

    rows: List[dict] = []

    # =============================
    # A. Generate 900 Controlled Samples
    # =============================
    # 450 true, 450 false (balanced)
    # 400 samples: pure rule-based (no paraphrase)
    # 500 samples: rule-based + LLM paraphrase (natural language)

    logging.info("=" * 70)
    logging.info("Phase 1: Generating 900 controlled samples")
    logging.info("  - 400 pure rule-based + 500 with LLM paraphrase")
    logging.info("=" * 70)
    print(
        "Generating 900 controlled samples (400 pure rule-based + 500 with LLM paraphrase)..."
    )
    controlled_samples = []

    for i in range(900):
        try:
            c = CONCEPTS[order[i % len(order)]]

            # Balanced: 450 true, 450 false
            is_true = i < 450

            # Generate evidence from concept
            evidence = make_evidence(c, rng, min_ev_fillers, max_ev_fillers)

            # Generate claim based on evidence (rule-based)
            if is_true:
                claim = make_true_claim(c, rng)
                label = "true"
            else:
                # For false samples in controlled set, mix contradiction and unsupported
                if rng.random() < 0.55:
                    claim = make_false_contradict_claim(c, rng)
                else:
                    claim = make_false_unsupported_claim(c, rng)
                label = "false"

            controlled_samples.append(
                {
                    "claim": claim,
                    "evidence": evidence,
                    "label": label,
                    "type": "controlled",
                }
            )

            # Log progress every 50 samples
            if (i + 1) % 50 == 0:
                logging.info(
                    f"  ✓ Generated {i + 1}/900 controlled samples ({(i + 1) / 900 * 100:.1f}%)"
                )
                print(
                    f"  Progress: {i + 1}/900 controlled samples ({(i + 1) / 900 * 100:.1f}%)"
                )

        except Exception as e:
            logging.error(f"Error generating controlled sample {i + 1}: {str(e)}")
            logging.warning(f"Skipping sample {i + 1} and continuing...")
            continue

    # Randomly select 500 samples for LLM paraphrase
    logging.info(
        "\nApplying LLM paraphrase to 500 randomly selected controlled samples..."
    )
    paraphrase_indices = set(rng.sample(range(900), 500))
    paraphrase_count = 0

    for idx, sample in enumerate(controlled_samples):
        if idx in paraphrase_indices:
            try:
                # Apply LLM paraphrase for natural language
                sample["evidence"] = llm_paraphrase_evidence(sample["evidence"])
                sample["claim"] = llm_paraphrase_claim(sample["claim"])
                sample["type"] = "controlled_paraphrased"
                paraphrase_count += 1

                # Log progress every 50 paraphrased samples
                if paraphrase_count % 50 == 0:
                    logging.info(
                        f"  ✓ Paraphrased {paraphrase_count}/500 samples ({paraphrase_count / 500 * 100:.1f}%)"
                    )
                    print(
                        f"  Progress: Paraphrased {paraphrase_count}/500 samples ({paraphrase_count / 500 * 100:.1f}%)"
                    )

            except Exception as e:
                logging.error(f"Error paraphrasing sample {idx}: {str(e)}")
                logging.warning(f"Keeping original text for sample {idx}")
                continue

    logging.info(
        f"✓ Phase 1 complete: Generated {len(controlled_samples)} controlled samples"
    )
    rows.extend(controlled_samples)

    # =============================
    # B. Generate 600 Hard Set Samples (LLM-BASED)
    # =============================
    # These are challenging samples with LLM generation
    # All are false (to balance with controlled set)

    logging.info("\n" + "=" * 70)
    logging.info("Phase 2: Generating 600 hard set samples (LLM-based)")
    logging.info("=" * 70)
    print("Generating 600 hard set samples (LLM-based)...")

    # 300 contradiction samples
    logging.info("Phase 2a: Generating 300 contradiction samples...")
    print("  - 300 contradiction samples...")
    for i in range(300):
        try:
            c = CONCEPTS[order[(900 + i) % len(order)]]

            # Generate evidence
            evidence = make_evidence(c, rng, min_ev_fillers, max_ev_fillers)

            # Paraphrase evidence for variety
            evidence = llm_paraphrase_evidence(evidence)

            # Generate contradictory claim using LLM
            claim = make_hard_contradiction(evidence, c.topic, rng)

            rows.append(
                {
                    "claim": claim,
                    "evidence": evidence,
                    "label": "false",
                    "type": "hard_contradiction",
                }
            )

            # Log progress every 50 samples
            if (i + 1) % 50 == 0:
                logging.info(
                    f"  ✓ Generated {i + 1}/300 contradiction samples ({(i + 1) / 300 * 100:.1f}%)"
                )
                print(
                    f"  Progress: {i + 1}/300 contradiction samples ({(i + 1) / 300 * 100:.1f}%)"
                )

        except Exception as e:
            logging.error(f"Error generating contradiction sample {i + 1}: {str(e)}")
            logging.warning(f"Skipping contradiction sample {i + 1} and continuing...")
            continue

    logging.info("✓ Phase 2a complete: Generated 300 contradiction samples")

    # 300 unsupported samples
    logging.info("\nPhase 2b: Generating 300 unsupported samples...")
    print("  - 300 unsupported samples...")
    for i in range(300):
        try:
            c = CONCEPTS[order[(1200 + i) % len(order)]]

            # Generate evidence
            evidence = make_evidence(c, rng, min_ev_fillers, max_ev_fillers)

            # Paraphrase evidence for variety
            evidence = llm_paraphrase_evidence(evidence)

            # Generate unsupported claim (adds absolute conditions)
            claim = make_hard_unsupported(evidence, c.topic, rng)

            rows.append(
                {
                    "claim": claim,
                    "evidence": evidence,
                    "label": "false",
                    "type": "hard_unsupported",
                }
            )

            # Log progress every 50 samples
            if (i + 1) % 50 == 0:
                logging.info(
                    f"  ✓ Generated {i + 1}/300 unsupported samples ({(i + 1) / 300 * 100:.1f}%)"
                )
                print(
                    f"  Progress: {i + 1}/300 unsupported samples ({(i + 1) / 300 * 100:.1f}%)"
                )

        except Exception as e:
            logging.error(f"Error generating unsupported sample {i + 1}: {str(e)}")
            logging.warning(f"Skipping unsupported sample {i + 1} and continuing...")
            continue

    logging.info("✓ Phase 2b complete: Generated 300 unsupported samples")

    # Shuffle all samples
    logging.info("\nShuffling all samples for randomization...")
    rng.shuffle(rows)
    logging.info(f"✓ All phases complete: Total {len(rows)} samples generated")
    logging.info("=" * 70)

    return rows


def write_csv(path: str, rows: List[dict]) -> None:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["claim", "evidence", "label", "type"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate 1500 high-quality test samples: 400 pure rule-based + 500 paraphrased + 600 LLM hard set"
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    ap.add_argument(
        "--out",
        type=str,
        default="./synthetic_finance_1500.csv",
        help="Output CSV file path",
    )
    ap.add_argument(
        "--log",
        type=str,
        default="./gen_data.log",
        help="Log file path for detailed logging",
    )
    args = ap.parse_args()

    # Configure logging
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Create file handler for detailed logging (INFO and above)
    file_handler = logging.FileHandler(args.log, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Create console handler for errors and warnings only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter(log_format))

    # Configure root logger
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    print("=" * 70)
    print("High-Quality Test Set Generation")
    print("=" * 70)
    print(
        "Structure: 400 pure rule-based + 500 paraphrased + 600 LLM hard set = 1500 total"
    )
    print(f"Seed: {args.seed}")
    print(f"Log file: {args.log}")
    print("=" * 70)
    print()

    logging.info("=" * 70)
    logging.info("Starting data generation process")
    logging.info(f"Seed: {args.seed}")
    logging.info(f"Output file: {args.out}")
    logging.info(f"Log file: {args.log}")
    logging.info("=" * 70)

    rows = build_rows(seed=args.seed)

    write_csv(args.out, rows)

    # Calculate statistics
    true_n = sum(1 for r in rows if r["label"] == "true")
    false_n = len(rows) - true_n

    controlled_pure = sum(1 for r in rows if r.get("type") == "controlled")
    controlled_para = sum(1 for r in rows if r.get("type") == "controlled_paraphrased")
    contradiction = sum(1 for r in rows if r.get("type") == "hard_contradiction")
    unsupported = sum(1 for r in rows if r.get("type") == "hard_unsupported")

    print()
    print("=" * 70)
    print("✓ Test set generation complete")
    print("=" * 70)
    print(f"Output: {args.out}")
    print(f"Total samples: {len(rows)}")
    print()
    print("Label distribution:")
    print(f"  - True:  {true_n} ({true_n / len(rows) * 100:.1f}%)")
    print(f"  - False: {false_n} ({false_n / len(rows) * 100:.1f}%)")
    print()
    print("Sample type distribution:")
    print(f"  - Controlled (pure rule-based):     {controlled_pure}")
    print(f"  - Controlled (with LLM paraphrase): {controlled_para}")
    print(f"  - Hard Contradiction (LLM):          {contradiction}")
    print(f"  - Hard Unsupported (LLM):            {unsupported}")
    print()
    print(f"Concepts used: {len(CONCEPTS)}")
    print("Columns: claim, evidence, label, type")
    print("=" * 70)
    
    # Log final statistics
    logging.info("")
    logging.info("=" * 70)
    logging.info("✓ Test set generation complete")
    logging.info("=" * 70)
    logging.info(f"Output file: {args.out}")
    logging.info(f"Total samples generated: {len(rows)}")
    logging.info("")
    logging.info("Label distribution:")
    logging.info(f"  - True:  {true_n} ({true_n / len(rows) * 100:.1f}%)")
    logging.info(f"  - False: {false_n} ({false_n / len(rows) * 100:.1f}%)")
    logging.info("")
    logging.info("Sample type distribution:")
    logging.info(f"  - Controlled (pure rule-based):     {controlled_pure}")
    logging.info(f"  - Controlled (with LLM paraphrase): {controlled_para}")
    logging.info(f"  - Hard Contradiction (LLM):         {contradiction}")
    logging.info(f"  - Hard Unsupported (LLM):           {unsupported}")
    logging.info("")
    logging.info(f"Concepts used: {len(CONCEPTS)}")
    logging.info("Columns: claim, evidence, label, type")
    logging.info("=" * 70)
    logging.info("Data generation process completed successfully!")


if __name__ == "__main__":
    main()
