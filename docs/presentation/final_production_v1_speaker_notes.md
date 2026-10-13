# Final Production V1 Speaker Notes

Use these as brief prompts while presenting `final_production_v1_meeting_deck.pdf`.

1. Title: This is the final analysis story, not a raw results dump.
2. Talk map: Emphasize the dependency chain: correctness first, privacy payoff last.
3. The problem: Releasing only the MLE compresses data; the question is what survives.
4. Models and regimes: Separate clean regimes from caveat regimes before showing results.
5. Frozen evidence source: `final_production_v1` anchors sampler claims; Step 4 now uses 100 datasets per regime.
6. Reference hierarchy: Raw weighted-MC is the posterior-summary benchmark; KDE smooths for display.
7. KDE stable example: Student k=3,n=20 is a clean visual walkthrough.
8. KDE sensitivity: Student k=1 should not be judged by a single smooth overlay.
9. Correctness standard: The pass condition is an evidence stack: posterior summaries, invariant target checks, and chain reliability.
10. Method-specific checks: Gibbs and RATTLE have different failure modes, so the diagnostics are sampler-specific.
11. Clean/caveat/unresolved: Clean is safe for the main comparison; targeted-validation upgrades carry multi-init/geometry evidence forward.
12. Reconciled verdict counts: Standalone final production had 11 caveats, but 9 are upgraded by targeted validation; only 2 caveats remain in the meeting-use view.
13. Posterior agreement: Clean rows agree; Student k=1,n=10 stays unresolved.
14. RATTLE checks: Projection/reverse/Hamiltonian diagnostics are the reason RATTLE is trustworthy where applicable.
15. Efficiency question: Compare cost per reliable posterior information.
16. Efficiency result: RATTLE wins ESS/sec in the clean comparable regimes.
17. Cost decomposition: In this implementation, Gibbs sec/iteration grows with n while RATTLE remains nearly flat.
18. Efficiency surface: The advantage grows in several larger-n smooth regimes.
19. Geometry matters: Gibbs exploits algebra locally; RATTLE uses geometric motion.
20. Student tail geometry: Heavy-tail latent states are the source of real difficulty.
21. Representative latent classes: Compare a clean Student k=3,n=20 case against unresolved Student k=1,n=10.
22. Sticky classes: This explains why caveats are substantive, not just cosmetic.
23. Parallel Cauchy follow-up: The deeper k=1 work is separate and should not be forced into the main comparison.
24. RATTLE near tails: Tail geometry stresses Hamiltonian behavior.
25. Final payoff: Now use trusted samplers to study information loss and leakage.
26. Normal sanity check: Known-variance normal is the zero-loss baseline.
27. Selected information loss: 100 datasets per regime; Student k=1,n=10 remains the standout loss case.
28. Heatmap: Loss depends on tail regime and n.
29. Privacy leakage: The MLE changes beliefs about latent extremes.
30. Compatible worlds: Same released statistic admits many latent configurations.
31. Dashboard walkthrough: Default to final production; keep caveats out of the default path.
32. Caveats: State boundaries clearly.
33. Takeaway: The project now has a full methodological and scientific arc.
