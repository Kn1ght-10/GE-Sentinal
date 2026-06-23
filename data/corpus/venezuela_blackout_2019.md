# Research note: Venezuela gold farming and the March 2019 blackouts

Press coverage in 2019 (including Slate and NPR's Planet Money) documented
that a meaningful share of OSRS gold farming was performed by players in
Venezuela earning real income from gold sales. In early March 2019, Venezuela
suffered multi-day nationwide power blackouts.

If Venezuelan farming was a large share of supply for farmable commodities,
the blackout is a natural experiment: an exogenous, sharply timed supply
shock. The testable prediction is abnormal positive returns in farmed
commodities versus a control basket during the blackout window, reverting as
power returned.

GE-Sentinel's causal module (event_study) is designed to run exactly this
comparison once daily history for the period is backfilled from the Weirdgloop
bulk archives. Verify exact blackout dates and the contemporaneous reporting
before publishing any result; this note is a research pointer, not a finding.
