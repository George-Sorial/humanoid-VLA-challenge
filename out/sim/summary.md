## Sim pick-and-place results

| strategy | layout shift | success | lifted | median place error | episodes |
|---|---|---|---|---|---|
| naive | 0 cm | **2/20** (10%) | 45% | 6.7 cm | 20 |
| naive | 3 cm | **0/20** (0%) | 10% | 7.0 cm | 20 |
| naive | 6 cm | **0/20** (0%) | 10% | 10.1 cm | 20 |
| anchored | 0 cm | **20/20** (100%) | 100% | 0.2 cm | 20 |
| anchored | 3 cm | **20/20** (100%) | 100% | 0.2 cm | 20 |
| anchored | 6 cm | **20/20** (100%) | 100% | 0.1 cm | 20 |
| **naive (all)** | | **2/60** | | | |
| **anchored (all)** | | **60/60** | | | |

## Per clip

| clip | your direction | marker spacing | hand frames | grasp err | release err | naive | anchored |
|---|---|---|---|---|---|---|---|
| L2R_1 | L→R | 34.9 cm | 82/123 | 1.7 cm | 3.4 cm | 1/3 | 3/3 |
| L2R_2 | L→R | 34.9 cm | 59/88 | 0.4 cm | 6.5 cm | 0/3 | 3/3 |
| L2R_3 | L→R | 35.0 cm | 61/94 | 1.7 cm | 5.2 cm | 0/3 | 3/3 |
| L2R_4 | L→R | 34.9 cm | 53/88 | 1.7 cm | 8.1 cm | 0/3 | 3/3 |
| L2R_5 | L→R | 34.9 cm | 48/93 | 1.4 cm | 7.7 cm | 0/3 | 3/3 |
| L2R_6 | L→R | 35.0 cm | 45/75 | 0.8 cm | 7.6 cm | 0/3 | 3/3 |
| L2R_7 | L→R | 35.1 cm | 40/74 | 1.4 cm | 5.9 cm | 0/3 | 3/3 |
| L2R_8 | L→R | 35.4 cm | 37/64 | 1.5 cm | 6.8 cm | 0/3 | 3/3 |
| L2R_9 | L→R | 35.7 cm | 37/69 | 0.3 cm | 4.1 cm | 0/3 | 3/3 |
| L2R_10 | L→R | 35.7 cm | 37/67 | 0.6 cm | 5.4 cm | 0/3 | 3/3 |
| R2L_1 | R→L | 35.8 cm | 28/54 | 4.8 cm | 0.3 cm | 0/3 | 3/3 |
| R2L_2 | R→L | 35.7 cm | 28/51 | 4.6 cm | 0.9 cm | 0/3 | 3/3 |
| R2L_3 | R→L | 35.8 cm | 31/47 | 4.8 cm | 2.0 cm | 0/3 | 3/3 |
| R2L_4 | R→L | 35.9 cm | 31/52 | 3.3 cm | 1.6 cm | 0/3 | 3/3 |
| R2L_5 | R→L | 35.8 cm | 25/49 | 2.7 cm | 0.4 cm | 0/3 | 3/3 |
| R2L_6 | R→L | 35.8 cm | 43/83 | 1.7 cm | 0.5 cm | 1/3 | 3/3 |
| R2L_7 | R→L | 35.9 cm | 30/54 | 2.2 cm | 2.0 cm | 0/3 | 3/3 |
| R2L_8 | R→L | 35.9 cm | 30/61 | 2.8 cm | 0.7 cm | 0/3 | 3/3 |
| R2L_9 | R→L | 35.9 cm | 30/62 | 4.8 cm | 1.8 cm | 0/3 | 3/3 |
| R2L_10 | R→L | 35.8 cm | 28/53 | 4.4 cm | 1.9 cm | 0/3 | 3/3 |

Success = cube lifted > 3 cm, then resting within 4 cm of the target. Grasp/release err = smoothed hand tip to object at the detected contact frame.
