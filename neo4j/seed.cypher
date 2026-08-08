// Bookly policy graph — reference Cypher.
//
// This is the same model that ingest.py writes, spelled out by hand so the
// shape is readable without running anything. Prefer:
//
//     python neo4j/ingest.py
//
// which reads policy_graph.json and is idempotent. Keep this file in step with
// that fixture; the two must not disagree.
//
// Neo4j holds policy only: item categories, return policies, return windows,
// exceptions, promotions, and regional overrides. Customers, orders, items, and
// returns live in ../data/*.json.

// --- Constraints ---------------------------------------------------------
CREATE CONSTRAINT category_name IF NOT EXISTS
  FOR (c:Category) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT policy_id IF NOT EXISTS
  FOR (p:Policy) REQUIRE p.policy_id IS UNIQUE;
CREATE CONSTRAINT region_code IF NOT EXISTS
  FOR (r:Region) REQUIRE r.code IS UNIQUE;

// --- Item categories -----------------------------------------------------
MERGE (physical:Category {name: 'PhysicalBook'})
  SET physical.description = 'Printed books. Returnable, subject to a window.';
MERGE (ebook:Category {name: 'EBook'})
  SET ebook.description = 'Digital downloads. Not returnable once made available.';

// --- Regions -------------------------------------------------------------
MERGE (au:Region {code: 'AU'}) SET au.name = 'Australia';
MERGE (gb:Region {code: 'GB'}) SET gb.name = 'United Kingdom';
MERGE (us:Region {code: 'US'}) SET us.name = 'United States';

// --- Policies ------------------------------------------------------------
// window_days null means returns are not offered at all — the absence of a
// window, not a window of zero length.
MERGE (standard:Policy {policy_id: 'STANDARD_30_DAY'})
  SET standard.name               = 'Standard 30-day returns',
      standard.window_days        = 30,
      standard.precedence         = 0,
      standard.window_starts_from = 'delivered_at',
      standard.exceptions         = ['DAMAGED_ON_ARRIVAL'],
      standard.summary            = 'Physical books may be returned within 30 days of delivery, in resalable condition.';

MERGE (digital:Policy {policy_id: 'DIGITAL_NO_RETURN'})
  SET digital.name        = 'Digital purchases are final',
      digital.window_days = null,
      digital.precedence  = 100,
      digital.exceptions  = [],
      digital.summary     = 'Ebooks cannot be returned once the download has been made available.';

MERGE (holiday:Policy {policy_id: 'HOLIDAY_EXTENDED_RETURN'})
  SET holiday.name                  = 'Holiday sale extended returns',
      holiday.window_days           = 60,
      holiday.precedence            = 5,
      holiday.window_starts_from    = 'delivered_at',
      holiday.promotion_code        = 'MIDYEAR_HOLIDAY_SALE_2026',
      holiday.promotion_active_from = '2026-06-15',
      holiday.promotion_active_to   = '2026-07-15',
      holiday.exceptions            = ['DAMAGED_ON_ARRIVAL'],
      holiday.summary               = 'Physical books bought during a Bookly holiday sale may be returned within 60 days of delivery instead of 30.';

// Bookly's own goodwill extension for Australia. Deliberately NOT framed as a
// statutory 45-day right — it is a Bookly policy and is named as one.
MERGE (aupolicy:Policy {policy_id: 'AU_BOOKLY_EXTENDED_RETURN'})
  SET aupolicy.name               = 'Bookly Australia extended returns',
      aupolicy.window_days        = 45,
      aupolicy.precedence         = 10,
      aupolicy.window_starts_from = 'delivered_at',
      aupolicy.exceptions         = ['DAMAGED_ON_ARRIVAL'],
      aupolicy.summary            = "A Bookly goodwill extension: customers in Australia get 45 days on physical books instead of 30. This is Bookly's own policy, not a statement about Australian law.";

// --- Which category is governed by which policy --------------------------
MATCH (c:Category {name: 'PhysicalBook'}), (p:Policy)
WHERE p.policy_id IN ['STANDARD_30_DAY', 'HOLIDAY_EXTENDED_RETURN', 'AU_BOOKLY_EXTENDED_RETURN']
MERGE (c)-[:GOVERNED_BY]->(p);

MATCH (c:Category {name: 'EBook'}), (p:Policy {policy_id: 'DIGITAL_NO_RETURN'})
MERGE (c)-[:GOVERNED_BY]->(p);

// --- Regional overrides --------------------------------------------------
MATCH (r:Region {code: 'AU'}), (p:Policy {policy_id: 'AU_BOOKLY_EXTENDED_RETURN'})
MERGE (r)-[:HAS_OVERRIDE]->(p);

// --- Precedence, as edges ------------------------------------------------
// "Which rule wins" is data, not if-statements scattered through tool code,
// and the traversal is what gets read back to the customer.
MATCH (h:Policy {policy_id: 'HOLIDAY_EXTENDED_RETURN'}), (s:Policy {policy_id: 'STANDARD_30_DAY'})
MERGE (h)-[:OVERRIDES]->(s);

MATCH (a:Policy {policy_id: 'AU_BOOKLY_EXTENDED_RETURN'}), (other:Policy)
WHERE other.policy_id IN ['STANDARD_30_DAY', 'HOLIDAY_EXTENDED_RETURN']
MERGE (a)-[:OVERRIDES]->(other);

// Nothing OVERRIDES DIGITAL_NO_RETURN, and it has no window. An ebook cannot be
// rescued by a region or a promotion, and that is expressed by absence.

// --- Verify --------------------------------------------------------------
// MATCH (a)-[r]->(b)
// RETURN a, r, b;

// --- Example reads (for the eligibility tool, not implemented yet) -------
//
// 1. Which policies could apply to a category for a customer in $country?
//
// MATCH (c:Category {name: $category})-[:GOVERNED_BY]->(p:Policy)
// WHERE NOT EXISTS { (:Region)-[:HAS_OVERRIDE]->(p) }
//    OR EXISTS { (:Region {code: $country})-[:HAS_OVERRIDE]->(p) }
// RETURN p ORDER BY p.precedence DESC;
//
// 2. The winning policy plus the rules it beat — the explainable path behind
//    EligibilityDecision.rule_path.
//
// MATCH (c:Category {name: $category})-[:GOVERNED_BY]->(p:Policy)
// WHERE NOT EXISTS { (:Region)-[:HAS_OVERRIDE]->(p) }
//    OR EXISTS { (:Region {code: $country})-[:HAS_OVERRIDE]->(p) }
// WITH p ORDER BY p.precedence DESC LIMIT 1
// OPTIONAL MATCH (p)-[:OVERRIDES]->(beaten:Policy)
// RETURN p.policy_id, p.window_days, collect(beaten.policy_id) AS overrides;
