#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 09:13:50 2025

@author: z3r0
"""

"""
mock_data.py
Simulated Output from the Finance Parser Agent.
Contains financial reports for Equity Group, CBK auction results, and SACCO sector data.
Updates: Replaced JSON 'null' with Python 'None'.
"""

# ============================================================================
# EQUITY GROUP HOLDINGS
# ============================================================================

EQUITY_GROUP_Q3 = {
    "meta": {
        "entity_legal_name": "Equity Group Holdings PLC",
        "doc_type": "Quarterly Financial Update (Q3)",
        "fiscal_period_end": "2025-09-30",
        "currency": "KES",
        "scale_detected": "Billions",
        "extraction_confidence": 0.95
    },
    "financials": {
        "income_statement": {
            "revenue": {
                "value": 156300000000.0,
                "raw_label": "Total Operating income"
            },
            "net_profit": {
                "value": 54100000000.0,
                "raw_label": "Profit after tax"
            },
            "operating_expenses": 90700000000.0
        },
        "balance_sheet": {
            "total_assets": 1817000000000.0,
            "total_liabilities": 1513800000000.0,
            "member_deposits": {
                "total": 1345500000000.0,
                "bosa_deposits": None,
                "fosa_deposits": None
            },
            "share_capital": None
        }
    },
    "sacco_metrics": {
        "proposed_dividend_rate": None,
        "proposed_interest_on_deposits_rate": None,
        "gross_loans": 859800000000.0,
        "npl_ratio": 13.6
    },
    "segments": [
        {
            "name": "Net Interest Income",
            "revenue": 93600000000.0
        },
        {
            "name": "Non-Funded Income",
            "revenue": 62700000000.0
        }
    ],
    "narrative": {
        "strategic_theme": "Strong earnings growth driven by increased operating income and improved asset quality, despite a challenging environment.",
        "risks": [
            "Elevated fuel prices compared to historical levels [cite: 1298]",
            "Persistent foreign investor outflows [cite: 1492]",
            "Cost of risk remains a factor, though improved to 7.4% [cite: 1450]"
        ]
    },
    "_notes": "Extracted from Cytonn Q3'2025 Earnings Note table[cite: 1449]. Share Capital explicitly not found in summary table; Shareholders Funds recorded at 288.4 Bn[cite: 1447]. NPL Ratio derived from text 'Gross decreasing to 13.6%'[cite: 1455]. Gross Loans field mapped to 'Net Loans and Advances' per schema fit, as Gross Loans (951.7 Bn) was available in text but 'Net Loans' (859.8 Bn) matches balance sheet line item[cite: 1447]."
}


# ============================================================================
# CENTRAL BANK OF KENYA - TREASURY BILL AUCTIONS
# ============================================================================

CBK_AUCTION_OCT = {
    "meta": {
        "entity_legal_name": "Central Bank of Kenya",
        "doc_type": "Auction Result",
        "fiscal_period_end": "2025-10-31",
        "currency": "KES",
        "scale_detected": "Absolute",
        "extraction_confidence": 0.98
    },
    "cbk_metrics": {
        "91_day_t_bill_rate": 7.9,
        "182_day_t_bill_rate": 7.9,
        "364_day_t_bill_rate": 9.4,
        "performance_rate": 0.976
    }
}


# ============================================================================
# CENTRAL BANK OF KENYA - BOND AUCTIONS
# ============================================================================

CBK_BOND_OCT_20 = {
    "meta": {
        "entity_legal_name": "Central Bank of Kenya",
        "doc_type": "Auction Result",
        "fiscal_period_end": "2025-10-20",
        "currency": "KES",
        "scale_detected": "Millions",
        "extraction_confidence": 0.99
    },
    "cbk_metrics": {
        "bond_auction": {
            "issue_numbers": ["FXD1/2018/015", "FXD1/2021/020"],
            "tenors": [7.7, 15.9],
            "amount_offered": 50000000000.0,
            "amount_received": 118900000000.0,
            "amount_accepted": 85300000000.0,
            "performance_rate": 2.378,
            "weighted_average_rates": [12.65, 13.53],
            "coupon_rates": [12.65, 13.44]
        }
    }
}

CBK_BOND_NOV = {
    "meta": {
        "entity_legal_name": "Central Bank of Kenya",
        "doc_type": "Auction Result",
        "fiscal_period_end": "2025-11-24",
        "currency": "KES",
        "scale_detected": "Millions",
        "extraction_confidence": 0.95
    },
    "cbk_metrics": {
        "bond_auction": {
            "issue_numbers": ["FXD3/2019/015", "FXD1/2022/025"],
            "tenors": [8.7, 21.9],
            "amount_offered": 40000000000.0,
            "amount_received": 115900000000.0,
            "amount_accepted": None,
            "performance_rate": 2.897,
            "coupon_rates": [12.34, 14.19]
        }
    },
    "_notes": "Auction highly oversubscribed (289.7%). Exact accepted amount not explicitly confirmed in summary snippet, often capped at offer or slightly higher."
}


# ============================================================================
# CENTRAL BANK OF KENYA - BUYBACK OPERATIONS
# ============================================================================

CBK_BUYBACK_NOV = {
    "meta": {
        "entity_legal_name": "Central Bank of Kenya",
        "doc_type": "Auction Result (Buyback)",
        "fiscal_period_end": "2025-11-19",
        "currency": "KES",
        "scale_detected": "Millions",
        "extraction_confidence": 0.99
    },
    "cbk_metrics": {
        "buyback_auction": {
            "issue_number": "FXD1/2023/003",
            "tenor_remaining": 0.6,
            "amount_offered_buyback": 30000000000.0,
            "amount_received": 34300000000.0,
            "amount_accepted": 20100000000.0,
            "performance_rate": 1.143,
            "weighted_average_yield": 7.80,
            "cut_off_price": 103.29
        }
    }
}


# ============================================================================
# SACCO SECTOR REPORTS
# ============================================================================

SACCO_SECTOR_23 = {
    "meta": {
        "entity_legal_name": "Regulated Sacco Subsector (Kenya)",
        "doc_type": "Sector Supervision Report",
        "fiscal_period_end": "2023-12-31",
        "currency": "KES",
        "scale_detected": "Billions",
        "extraction_confidence": 0.98
    },
    "financials": {
        "balance_sheet": {
            "total_assets": 971960000000.0,
            "total_liabilities": None,
            "member_deposits": {
                "total": 682000000000.0,
                "bosa_deposits": None,
                "fosa_deposits": None
            },
            "share_capital": None
        }
    },
    "sacco_metrics": {
        "gross_loans": 758000000000.0,
        "npl_ratio": 7.5,
        "proposed_dividend_rate": None,
        "proposed_interest_on_deposits_rate": None
    }
}

SACCO_SECTOR_24 = {
    "meta": {
        "entity_legal_name": "Regulated Sacco Subsector (Kenya)",
        "doc_type": "Sector Supervision Report",
        "fiscal_period_end": "2024-12-31",
        "currency": "KES",
        "scale_detected": "Trillions/Billions",
        "extraction_confidence": 0.95
    },
    "financials": {
        "balance_sheet": {
            "total_assets": 1080000000000.0,
            "total_liabilities": None,
            "member_deposits": {
                "total": 749000000000.0,
                "bosa_deposits": None,
                "fosa_deposits": None
            },
            "share_capital": None
        }
    },
    "sacco_metrics": {
        "gross_loans": 845000000000.0,
        "npl_ratio": 8.2,
        "proposed_dividend_rate": None,
        "proposed_interest_on_deposits_rate": None
    },
    "narrative": {
        "strategic_theme": "Resilient expansion with total assets crossing the Ksh 1 Trillion mark; focus on consolidation of smaller BOSA Saccos.",
        "risks": [
            "Rising Non-Performing Loans (NPLs) in agriculture and informal trade",
            "Cybersecurity threats",
            "Liquidity constraints due to delayed remittances from public sector"
        ]
    }
}


# ============================================================================
# REGISTRY - Quick lookup dictionary
# ============================================================================

FINANCIAL_DATA_REGISTRY = {
    "EQTY": EQUITY_GROUP_Q3,
    "CBK_TBILL": CBK_AUCTION_OCT,
    "CBK_BOND_OCT": CBK_BOND_OCT_20,
    "CBK_BOND_NOV": CBK_BOND_NOV,
    "CBK_BUYBACK": CBK_BUYBACK_NOV,
    "SACCO_2023": SACCO_SECTOR_23,
    "SACCO_2024": SACCO_SECTOR_24
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_report(ticker: str):
    """
    Retrieve financial report by ticker symbol.
    
    Args:
        ticker: Ticker symbol or report ID (e.g., "EQTY", "CBK_TBILL")
    
    Returns:
        Dictionary containing financial data, or None if not found
    """
    return FINANCIAL_DATA_REGISTRY.get(ticker.upper())


def list_available_reports():
    """List all available financial reports."""
    return list(FINANCIAL_DATA_REGISTRY.keys())