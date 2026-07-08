-- FILE 2: EXPORT BRAND ALIAS -> JSONL
-- doc_id = b_<rxaui>, text = str brand, role = "alias"
-- resolve_in  = list RXCUI của generic IN  (suppress='N')
-- resolve_min = list RXCUI của generic MIN (suppress='N')
--
-- Brand giữ CẢ suppress N lẫn O (bắt đơn thuốc cũ),
-- target resolve LUÔN suppress='N' (hiện hành).
--
-- KIẾN TRÚC: RXNREL cho mã (rxcui2 CHÍNH LÀ mã xuất ra),
--            RXNCONSO chỉ GÁC CỔNG (EXISTS: xác nhận rxcui2
--            có atom TTY=IN/MIN + suppress='N').
--
-- Flow:
--   brand row (rxcui_brand)
--     -> RXNREL: rxcui1 = rxcui_brand  => rxcui2 (generic, = mã xuất)
--     -> RXNCONSO EXISTS: rxcui2 có IN/MIN hiện hành thì mới giữ

WITH
-- brand concepts: BN + họ SBD (giữ cả suppress N và O)
brand AS (
    SELECT DISTINCT
        b.rxcui,
        b.rxaui,
        b.str
    FROM rxnconso b
    WHERE b.sab = 'RXNORM'
      AND b.tty IN ('BN', 'SBD', 'SBDC', 'SBDF', 'SBDG')
),
-- nối brand -> generic rxcui2 qua RXNREL (cấp RXCUI)
rel AS (
    SELECT DISTINCT
        br.rxaui   AS brand_rxaui,
        br.str     AS brand_str,
        r.rxcui2   AS generic_rxcui
    FROM brand br
    JOIN rxnrel r
      ON r.rxcui1 = br.rxcui
     AND r.sab = 'RXNORM'
),
-- resolve_in: mã = rxcui2 (lấy thẳng), RXNCONSO chỉ gác cổng qua EXISTS
resolve_in AS (
    SELECT DISTINCT
        rel.brand_rxaui,
        rel.generic_rxcui AS in_rxcui
    FROM rel
    WHERE EXISTS (
        SELECT 1 FROM rxnconso g
        WHERE g.rxcui = rel.generic_rxcui
          AND g.sab = 'RXNORM'
          AND g.tty = 'IN'
          AND g.suppress = 'N'
    )
),
-- resolve_min: y hệt, gác cổng bằng TTY='MIN'
resolve_min AS (
    SELECT DISTINCT
        rel.brand_rxaui,
        rel.generic_rxcui AS min_rxcui
    FROM rel
    WHERE EXISTS (
        SELECT 1 FROM rxnconso g
        WHERE g.rxcui = rel.generic_rxcui
          AND g.sab = 'RXNORM'
          AND g.tty = 'MIN'
          AND g.suppress = 'N'
    )
)
SELECT
    json_build_object(
        'doc_id',      'b_' || br.rxaui,
        'text',        br.str,
        'role',        'alias',
        'resolve_in',  COALESCE(
                          (SELECT json_agg(DISTINCT ri.in_rxcui)
                           FROM resolve_in ri
                           WHERE ri.brand_rxaui = br.rxaui),
                          '[]'::json
                       ),
        'resolve_min', COALESCE(
                          (SELECT json_agg(DISTINCT rm.min_rxcui)
                           FROM resolve_min rm
                           WHERE rm.brand_rxaui = br.rxaui),
                          '[]'::json
                       )
    ) AS jsonl_line
FROM brand br
WHERE EXISTS (SELECT 1 FROM resolve_in  ri WHERE ri.brand_rxaui = br.rxaui)
   OR EXISTS (SELECT 1 FROM resolve_min rm WHERE rm.brand_rxaui = br.rxaui)
ORDER BY br.rxaui;