-- FILE 1: EXPORT CANDIDATE (generic IN) -> JSONL
-- doc_id = c_<rxcui>, text = str của IN, code = RXCUI,
-- role = "candidate", synonyms = list str cùng rxcui (SY/TMSY/...)
-- Lọc: sab='RXNORM', tty='IN', suppress='N'

WITH
-- các generic IN đủ điều kiện làm candidate (định danh theo RXCUI)
candidate_in AS (
    SELECT DISTINCT
        c.rxcui,
        c.str
    FROM rxnconso c
    WHERE c.sab = 'RXNORM'
      AND c.tty IN ('IN', 'SCD')
      AND c.suppress = 'N'
),
-- gom synonyms cùng rxcui: các cách viết khác trong RXNORM (SY, TMSY)
syns AS (
    SELECT
        ci.rxcui   AS in_rxcui,
        s.str      AS syn_str
    FROM candidate_in ci
    JOIN rxnconso s
      ON s.rxcui = ci.rxcui
     AND s.sab   = 'RXNORM'
     AND s.tty IN ('SY', 'TMSY')
     AND s.str <> ci.str
)
SELECT
    json_build_object(
        'doc_id',   'c_' || ci.rxcui,
        'text',     ci.str,
        'code',     ci.rxcui,
        'role',     'candidate',
        'synonyms', COALESCE(
                        (SELECT json_agg(DISTINCT sy.syn_str)
                         FROM syns sy
                         WHERE sy.in_rxcui = ci.rxcui),
                        '[]'::json
                    )
    ) AS jsonl_line
FROM candidate_in ci
ORDER BY ci.rxcui;
