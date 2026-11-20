# Findings before a new learned planner

2026-11-20. Six independent investigations of the planner problem, made before any new
training, and what follows from them. This file is the starting point for the learned
planner; `docs/systems.md` describes what is built.

All numbers: `scenarios/demo.npz`, the settings of `experiments/eval_demo.toml` as they were
that day (40 drones of 40 L, one station in the corner, two random fires of 20 cells, at
most 64 targets on offer), validation seeds, simulator digest 855306543563. Area saved is
the metric that leads; fires put out is reported beside it. The probe scripts lived in the
session's scratch directory and are not kept; every claim below that matters is to be
re-measured by the tools in the repository.

## What was found

1. **Greedy and rollout are weak baselines.** A rule with three numbers (never send a drone
   to a block whose strongest front is above 1.5 MW/m; among the rest take the lowest of
   `distance in m + 100 x intensity in MW/m + 10 x open drops`) saved 54.9 % of the area on
   500 seeds, against 43.4 % for greedy and 46.8 % for rollout (300 s, 4 candidates). Tuned
   on seeds 0-249, 55.1 % on seeds 250-499. A second investigation found the same with its
   own probe (+6 to +9 points over greedy). The gain is almost all on shrub fires (above
   2000 kW/m when the drones start: 44 % against 27 %), by leaving intense fronts dry.
2. **The task is triage.** The pads deliver about 8 drops a minute (8 pads, 60 s) against
   an open need of about 240 drops; 45 % of the blocks on offer are above 1.5 MW/m and hold
   69 % of the open need. Ready drones wait 1.4 % of the time: there is always something to
   drop on, the question is what to skip.
3. **The best cap depends on the team**: 1.0 MW/m at 20 drones (29.6 % against greedy's
   24.7 %), 1.5 at 40, 2.5 at 80 (76.2 % against 66.9 %). The 40-drone weights gained
   nothing significant at 20 drones.
4. **Rollout is limited by its candidates, not its horizon.** 600 and 900 s add nothing;
   150 s loses. With 8, 16, 32, 64 candidates it saved 50.0, 52.0, 53.4, 55.2 % on 100 seeds
   (48.2 % with 4) and had not levelled off. Its candidates are greedy's choice, waiting and
   the *neediest* blocks, which are the most intense: the opposite of what pays.
5. **Rollout is a poor teacher.** It departs from greedy on 2.6 to 4.5 % of drone decisions
   (about 11 to 14 per episode); each is worth about 1.5 cells at 300 s; a conditional logit
   on all 41 features tried predicted 0 of 377 held-out departures. A network that imitates
   it is capped at its level, below the rule.
6. **What a single decision changes is small against what a fire changes.** Paired between
   planners that work differently the standard deviation is about 15 points of area saved
   per episode (7 between greedy and rollout): 500 seeds resolve about 1.3 points, 0.3
   points need about 10,000. Results of 100 seeds below carry that weight.
7. **Docked drones are interchangeable.** 95 % of decisions are for drones docked at the
   station, with identical tank, position and mask; shuffling rollout's choices among them
   costs 0.011 of its 0.205 points of burned area (not significant). A median of 2 drones
   decide per event (40 at the start), about 90 events and 300 to 400 drone decisions per
   episode. In 44 % of drone decisions waiting is forced (nothing open in reach).
8. **The offer of 64 targets binds** in 28 to 44 % of decision events, and it keeps the
   neediest blocks. Without it greedy burns 0.05 points less (+0.8 points of area saved
   with 128 slots); rollout gains the same.
9. **Where the station stands matters more than the planner**: in the map centre greedy
   saves 64.9 % (fire out 43 %), in the corner 43.4 % (17 %). The owner keeps the corner
   station: stations are fixed installations (for example water points in a forest), and
   where one would stand is not ours to choose.
10. **On other layouts the ranking changes** (100 seeds each): on `haard`, `baggersee` and
    the demo map with a central station, `flank` beat rollout by 3 to 6 points of area saved
    while rollout put out more fires. One map is not enough to judge a planner.
11. **Features.** What pays is mostly what the planner is already shown: intensity and need.
    On top of them, measured as terms of a greedy-speed score:
    - open neighbouring blocks, fewer is better (a drop in the middle of an open stretch is
      undone when its neighbours re-ignite it): +1.0 to +1.6 points; with the wrong sign -20;
    - skipping fires of whose fronts more than 30 % are beyond the water limit: +0.8 points
      and 6 % fewer drops; the only rule that beat rollout at 20 drones (exploratory);
    - supply against demand (ready drones, pad queue, open need, drops per minute): not
      measured as a feature, inferred from finding 3;
    - no value or harmful as linear terms: head or flank position in the wind frame (-3.7),
      what a no-action forecast says is at stake (-3.4), water lying on the block (-2.5 to
      -5.7), front age (down to -10.9), front cells, distance home, fuel ahead beyond the
      limit, spread rate (area 0, more fires out).
    - a minimum open need of one drop: -16 (fractional remainders must be covered); holding
      docked drones back: 26.7 % area saved.
12. **Under the belief** (4 explorers, 2 tracking, forecast, 100 seeds): rollout whose
    lookahead plays in the forecast gains +1.8 points over greedy with fires reported within
    100 m and nothing without a report (+4.3 with full knowledge on the same seeds). Rollout
    on the belief *without* a forecast is worse than greedy (-1.5): its lookahead assumes
    later decisions see the true fire.
13. **The earlier PPO result was not a fair test** (old task, 4 drones):
    - trained and logged as a sampling policy, scored by argmax; its sampled policy beat the
      imitation's sampled policy on the same training seeds (5.7 % against 6.4 % burned);
    - the learning rate reached zero while it still improved; one seed; no evaluation
      during training, only the last weights kept;
    - one clipping ratio per joint action of up to 40 drones: steps with 10 or more deciders
      (35 % of all decisions) were clipped in 15 to 44 % of samples; entropy summed over
      deciders and not clipped;
    - the value head trains the encoder the policy shares: four critic-only updates moved
      the imitated policy more than four PPO updates; fine-tuning started with an untrained
      critic and nothing anchored it (the collapse to "keep");
    - the time limit bootstraps from a value that is never trained;
    - the default `[ppo]` of `experiments/planner.toml` would need about 34 GiB per
      minibatch; `PlannerNet` loops over all drones in Python (55 to 80 % of collection);
    - unscaled inputs (need 0-38, inbound 0-19, `task_kind` as one number, x and y as
      fractions of different lengths); `imitation.pt` is saved as schema 1 and refused.
    - Sound and to be kept: the re-scoring path (exact), the reward (the objective itself),
      the discount per second, the end-of-episode handling, experiment files, the
      evaluation runner, the DAgger loop.
    - The genuine problem of the method: 40 drones share one reward, and with a learned
      state value as the only baseline the signal of a decision is about 1/40 of the spread
      between fires.
14. **After reset an episode is a function of the actions** (Rothermel, `ros_noise` 0): for
    any deterministic base policy the value of an option is exact from one simulation.
15. **Ceiling**, estimated: about 60 % area saved on this benchmark (58 to 63 %). Grass
    fires look used up at about 79 %; shrub fires have room (rule 44 %, best of all arms per
    episode 50 %).

## Built since (2026-11-20)

The offer was raised to 256 targets, the rule became the `triage` planner with a fourth
number (open neighbours) tuned by `python -m roshan.tune`, and the lookahead can run on top
of it (`triage_rollout`). On 500 validation seeds with 40 drones: triage saves 57.0 % of the
area on the demo map and 59.9 % on `haard_station` (not tuned there), triage with lookahead
64.7 % and 64.1 % (65.7 % and 65.1 % with 16 candidates); greedy 44.2 % and 50.6 %. That is
above the ceiling estimated in finding 15, which was made with 64 targets and without the
lookahead on the rule. The best cap grows with the team (about 1250, 1600, 2750 kW/m at 20,
40, 80 drones); the 40-drone numbers lose about a point at 20 and 80. Tables:
`docs/systems.md`, "Results", "Triage".

So the bar for a learned planner is `triage` at greedy's cost (57 %) and `triage_rollout`
with search (65 %). The open question from finding 14 can now be measured: whether a learned
policy as the base of the lookahead compounds beyond 65 %.

## What follows for a learned planner

- **Baseline to beat:** the tuned triage rule and the lookahead on top of it, on several
  team sizes and more than one map, not greedy and rollout.
- **What it is for:** what a fixed rule cannot do. Adapting the triage to the team and the
  situation (finding 3), transferring across maps and layouts (10), planning under the
  belief (12).
- **Formulation:** "where does the next drop go" per station, one pick after another with
  the drops already claimed as an input (a count, not a flag); waiting ends a station's
  picks; drones in the air are left to the rule at first. A static ranking of blocks filled
  to the open need can express only 42 % of rollout's departures; sending more than the open
  need must stay possible (forbidding it costs rollout about a quarter of its gain).
- **Network:** a shared scorer per block with pooled context (mean and max over blocks,
  per-fire pools, globals) and a pair head; about 40,000 parameters; 20 to 40 microseconds
  per event in plain Rust, so it can be the base policy inside a lookahead. No recurrent
  state: the belief is the memory. The critic, if any, is a separate and larger network.
- **Inputs:** intensity, need, open after claims, open neighbours, a fire table (share of
  fronts beyond the limit, open need, growth), supply against demand, flight seconds instead
  of map fractions, vectors in the wind frame relative to the fire and the station. No map.
- **Training, first choice:** label each decision with the simulated outcome of its options
  (the rule or the network playing on afterwards), train the scores on those, then use the
  learned policy as the base of the lookahead and repeat. Before investing, measure whether
  one step of improvement compounds: lookahead on top of lookahead on 100 seeds.
- **Training, second choice:** black-box search (cross-entropy method, CMA-ES) over the
  weights of a small scorer on the true objective with the same seeds for every candidate.
- **PPO** only as a comparison arm, with: one ratio and clip per drone decision, entropy
  per decision, a separate critic encoder, same-seed groups as baseline, the time limit as
  an ending, the network run on deciding drones only, evaluation by argmax during training
  with the best checkpoint kept, three seeds or more.
- **Evaluation:** thousands of seeds outside the splits during development, the 500
  validation seeds for the final check, results by the fire's intensity at the start, area
  saved and fires out side by side, paired intervals, time per decision.

## Other directions noted, not pursued

- Sending attack drones towards a reported fire before the first sighting: bounded by
  +1.4 points with 100 m reports, +6.7 with 300 m reports.
- More pads (+3.9 points for 16) and larger tanks are hardware, not planning.
- Retargeting in flight: at most part of what reacting is worth (3 to 4 points).
- Indirect attack (wet lines ahead of the front): the literature treats water as a
  direct-attack agent; any result here would rest on the assumed `water_tau_s`.
- Mixed fleets would turn the assignment into a real optimisation problem.

## Sources named by the investigations

Bertsekas 2022 (rollout and policy iteration as a Newton step); Bhattacharya et al. 2020,
arXiv 2011.04222 (multiagent rollout with policy networks); Anthony et al. 2017, arXiv
1705.08439 (expert iteration); Ross & Bagnell 2014, arXiv 1406.5979 (AggreVaTe); Gabillon et
al. 2013 (classification-based policy iteration on Tetris); Szita & Lorincz 2006
(cross-entropy method on Tetris); Kool et al. 2019, arXiv 1803.08475 (attention model for
routing); Zhang et al. 2020, arXiv 2010.12367 (learned dispatching rules); Haksar & Schwager
2018 (distributed deep RL for fighting forest fires); Alexander 2000 (fire behaviour as a
factor in suppression); Plucinski et al. 2012 (aerial suppression and initial attack); Fried
& Fried 1996 (containment). Not re-checked here.
