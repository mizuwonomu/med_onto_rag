WITH params AS (
    SELECT '902029'::varchar AS brand_rxaui
),

-- B1: từ rxaui brand -> lấy thông tin brand (rxcui, str, tty, suppress)
brand AS (
    SELECT b.rxaui, b.rxcui, b.tty, b.suppress, b.str
    FROM rxnconso b
    JOIN params p ON b.rxaui = p.brand_rxaui
),

-- B2: join RXNREL rxcui1(brand) -> rxcui2(generic)
rel AS (
    SELECT DISTINCT
        br.rxaui AS brand_rxaui,
        br.str   AS brand_str,
        r.rela,
        r.rxcui2 AS generic_rxcui
    FROM brand br
    JOIN rxnrel r
      ON r.rxcui1 = br.rxcui
     AND r.sab = 'RXNORM'
)

-- B3: gác cổng qua RXNCONSO, hiện đủ thông tin generic để soi
SELECT
    rel.brand_rxaui,
    rel.brand_str,
    rel.rela,
    rel.generic_rxcui,
    g.tty        AS generic_tty,
    g.suppress   AS generic_suppress,
    g.str        AS generic_str
FROM rel
JOIN rxnconso g
  ON g.rxcui = rel.generic_rxcui
 AND g.sab = 'RXNORM'
 AND g.tty IN ('IN', 'MIN')
 AND g.suppress = 'N'
ORDER BY g.tty, rel.generic_rxcui;