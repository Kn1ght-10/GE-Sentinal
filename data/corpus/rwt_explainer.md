# Real-world trading (RWT) on the Grand Exchange — sample corpus note

Real-world trading means exchanging in-game gold or items for actual money,
which is prohibited by the game's rules. Jagex enforcement targets both sides
of the trade: accounts that sell gold and accounts that buy it have both been
banned in enforcement actions.

A pattern the OSRS Wiki team has described in its real-time prices FAQ is that
suspected RWT transfers can show up in market data as low-volume items being
traded at extraordinary prices: the "price" is really a disguised transfer of
value between two parties rather than a market-clearing trade. GE-Sentinel's
rmt_spike signature is built directly on this idea — a low-liquidity item
printing at many multiples of its rolling median on one or two units of
volume, often one-sided.

Because the underlying real-money gold market prices gold at a small number
of dollars per million coins, very large transfers tend to leave detectable
footprints when routed through illiquid items.
