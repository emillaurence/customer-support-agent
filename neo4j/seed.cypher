// Bookly policy graph — illustrative seed.
//
// Nothing runs this yet. It documents the exact shape that
// policy_graph.json stands in for; the two files must stay consistent.
//
// Neo4j holds policy only: product categories, return policies, return
// windows, exceptions, promotions, and regional overrides. Customers,
// orders, items, and returns live in ../data/*.json.

// --- Constraints ---------------------------------------------------------
CREATE CONSTRAINT product_type_name IF NOT EXISTS
  FOR (t:ProductType) REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT region_code IF NOT EXISTS
  FOR (r:Region) REQUIRE r.code IS UNIQUE;
CREATE CONSTRAINT policy_id IF NOT EXISTS
  FOR (p:Policy) REQUIRE p.policy_id IS UNIQUE;
CREATE CONSTRAINT return_window_id IF NOT EXISTS
  FOR (w:ReturnWindow) REQUIRE w.window_id IS UNIQUE;
CREATE CONSTRAINT promotion_code IF NOT EXISTS
  FOR (pr:Promotion) REQUIRE pr.code IS UNIQUE;
CREATE CONSTRAINT exception_code IF NOT EXISTS
  FOR (e:Exception) REQUIRE e.code IS UNIQUE;

// --- Product categories --------------------------------------------------
MERGE (physical:ProductType {name: 'PhysicalBook'})
  SET physical.returnable = true;
MERGE (ebook:ProductType {name: 'EBook'})
  SET ebook.returnable = false;

// --- Regions -------------------------------------------------------------
MERGE (au:Region {code: 'AU'}) SET au.name = 'Australia';
MERGE (gb:Region {code: 'GB'}) SET gb.name = 'United Kingdom';
MERGE (us:Region {code: 'US'}) SET us.name = 'United States';

// --- Return windows ------------------------------------------------------
// Windows are nodes, not properties, so an explanation can name the window
// it measured against: PhysicalBook -> STANDARD_30_DAY -> 30 days.
MERGE (w30:ReturnWindow {window_id: 'WINDOW_30_DAY'})
  SET w30.days = 30, w30.starts_from = 'delivered_at';
MERGE (w45:ReturnWindow {window_id: 'WINDOW_45_DAY'})
  SET w45.days = 45, w45.starts_from = 'delivered_at';
MERGE (w60:ReturnWindow {window_id: 'WINDOW_60_DAY'})
  SET w60.days = 60, w60.starts_from = 'delivered_at';

// --- Policies ------------------------------------------------------------
MERGE (standard:Policy {policy_id: 'STANDARD_30_DAY'})
  SET standard.name       = 'Standard 30-day returns',
      standard.precedence = 0,
      standard.summary    = 'Physical books may be returned within 30 days of delivery, in resalable condition.';

MERGE (digital:Policy {policy_id: 'DIGITAL_NO_RETURN'})
  SET digital.name       = 'Digital purchases are final',
      digital.precedence = 100,
      digital.summary    = 'Ebooks cannot be returned once the download has been made available.';

MERGE (holiday:Policy {policy_id: 'HOLIDAY_EXTENDED_RETURN'})
  SET holiday:PromotionalPolicy,
      holiday.name       = 'Holiday sale extended returns',
      holiday.precedence = 5,
      holiday.summary    = 'Physical books bought during a Bookly holiday sale may be returned within 60 days of delivery instead of 30.';

// Bookly's own goodwill extension for Australia. Deliberately NOT framed as a
// statutory 45-day right — it is a Bookly policy and is named as one.
MERGE (aupolicy:Policy {policy_id: 'AU_BOOKLY_EXTENDED_RETURN'})
  SET aupolicy:RegionalPolicy,
      aupolicy.name       = 'Bookly Australia extended returns',
      aupolicy.precedence = 10,
      aupolicy.summary    = "A Bookly goodwill extension: customers in Australia get 45 days on physical books instead of 30. This is Bookly's own policy, not a statement about Australian law.";

// --- Promotions ----------------------------------------------------------
MERGE (promo:Promotion {code: 'MIDYEAR_HOLIDAY_SALE_2026'})
  SET promo.name        = 'Mid-year holiday sale 2026',
      promo.active_from = date('2026-06-15'),
      promo.active_to   = date('2026-07-15');

// --- Exceptions ----------------------------------------------------------
MERGE (damaged:Exception {code: 'DAMAGED_ON_ARRIVAL'})
  SET damaged.name    = 'Damaged or faulty on arrival',
      damaged.summary = 'A book that arrived damaged or faulty is returnable even if the window has closed. Needs human review before approval.';

// --- Which category is governed by which policy --------------------------
MATCH (t:ProductType {name: 'PhysicalBook'}), (p:Policy)
WHERE p.policy_id IN ['STANDARD_30_DAY', 'HOLIDAY_EXTENDED_RETURN', 'AU_BOOKLY_EXTENDED_RETURN']
MERGE (t)-[:GOVERNED_BY]->(p);

MATCH (t:ProductType {name: 'EBook'}), (p:Policy {policy_id: 'DIGITAL_NO_RETURN'})
MERGE (t)-[:GOVERNED_BY]->(p);

// --- Which policy measures against which window --------------------------
// DIGITAL_NO_RETURN gets no HAS_WINDOW edge. The missing edge IS the rule.
MATCH (p:Policy {policy_id: 'STANDARD_30_DAY'}), (w:ReturnWindow {window_id: 'WINDOW_30_DAY'})
MERGE (p)-[:HAS_WINDOW]->(w);

MATCH (p:Policy {policy_id: 'AU_BOOKLY_EXTENDED_RETURN'}), (w:ReturnWindow {window_id: 'WINDOW_45_DAY'})
MERGE (p)-[:HAS_WINDOW]->(w);

MATCH (p:Policy {policy_id: 'HOLIDAY_EXTENDED_RETURN'}), (w:ReturnWindow {window_id: 'WINDOW_60_DAY'})
MERGE (p)-[:HAS_WINDOW]->(w);

// --- Regional overrides --------------------------------------------------
MATCH (r:Region {code: 'AU'}), (p:Policy {policy_id: 'AU_BOOKLY_EXTENDED_RETURN'})
MERGE (r)-[:HAS_OVERRIDE]->(p);

// --- Promotional grants --------------------------------------------------
MATCH (pr:Promotion {code: 'MIDYEAR_HOLIDAY_SALE_2026'}), (p:Policy {policy_id: 'HOLIDAY_EXTENDED_RETURN'})
MERGE (pr)-[:GRANTS]->(p);

// --- Precedence, as edges ------------------------------------------------
// "Which rule wins" is data, not if-statements scattered through tool code,
// and the traversal is what gets read back to the customer.
MATCH (h:Policy {policy_id: 'HOLIDAY_EXTENDED_RETURN'}), (s:Policy {policy_id: 'STANDARD_30_DAY'})
MERGE (h)-[:OVERRIDES]->(s);

MATCH (a:Policy {policy_id: 'AU_BOOKLY_EXTENDED_RETURN'}), (other:Policy)
WHERE other.policy_id IN ['STANDARD_30_DAY', 'HOLIDAY_EXTENDED_RETURN']
MERGE (a)-[:OVERRIDES]->(other);

// Nothing OVERRIDES DIGITAL_NO_RETURN. An ebook cannot be rescued by region
// or promotion, and that is expressed by the absence of an edge.

// --- Exception waivers ---------------------------------------------------
MATCH (e:Exception {code: 'DAMAGED_ON_ARRIVAL'}), (p:Policy)
WHERE p.policy_id IN ['STANDARD_30_DAY', 'AU_BOOKLY_EXTENDED_RETURN', 'HOLIDAY_EXTENDED_RETURN']
MERGE (e)-[:WAIVES]->(p);

// A faulty ebook is an escalation, not a return — so DAMAGED_ON_ARRIVAL
// deliberately does not WAIVE DIGITAL_NO_RETURN.

// --- Example reads (for the eligibility tool, Phase 3) -------------------
//
// 1. Which policies could apply to a physical book for an AU customer?
//
// MATCH (t:ProductType {name: $product_type})-[:GOVERNED_BY]->(p:Policy)
// WHERE NOT p:RegionalPolicy
//    OR (:Region {code: $country})-[:HAS_OVERRIDE]->(p)
// RETURN p ORDER BY p.precedence DESC;
//
// 2. The winning policy plus its window and the rules it beat — the
//    explainable path behind EligibilityDecision.rule_path.
//
// MATCH (t:ProductType {name: $product_type})-[:GOVERNED_BY]->(p:Policy)
// WHERE NOT p:RegionalPolicy
//    OR (:Region {code: $country})-[:HAS_OVERRIDE]->(p)
// WITH p ORDER BY p.precedence DESC LIMIT 1
// OPTIONAL MATCH (p)-[:HAS_WINDOW]->(w:ReturnWindow)
// OPTIONAL MATCH (p)-[:OVERRIDES]->(beaten:Policy)
// RETURN p.policy_id, w.days, collect(beaten.policy_id) AS overrides;
//
// 3. Is a promotional policy actually live for this delivery date?
//
// MATCH (pr:Promotion)-[:GRANTS]->(p:Policy {policy_id: $policy_id})
// WHERE $delivered_at >= pr.active_from AND $delivered_at <= pr.active_to
// RETURN pr.code;
