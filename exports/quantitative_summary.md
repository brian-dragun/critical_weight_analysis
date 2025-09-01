# Critical Weight Analysis - Quantitative Summary

## Sensitivity Score Distribution Analysis

### Top 10 Most Critical Weights
1. **0.330103** - Highest sensitivity (embedding layer)
2. **0.271084** - Second highest 
3. **0.269435** - Third highest
4. **0.252308** - Fourth highest
5. **0.197902** - Fifth highest
6. **0.170183** - Sixth highest
7. **0.154561** - Seventh highest
8. **0.146319** - Eighth highest
9. **0.145552** - Ninth highest
10. **0.144135** - Tenth highest

### Statistical Summary
- **Maximum Sensitivity:** 0.330103 (33% impact)
- **Minimum Sensitivity:** 0.000188 (0.02% impact)
- **Mean Sensitivity:** 0.002271 (0.23% average impact)
- **Median Sensitivity:** 0.001336 (0.13% median impact)

### Key Insights
- **Extreme Concentration:** Top sensitivity is 1,756x higher than minimum
- **Heavy-Tailed Distribution:** Few weights have extremely high sensitivity
- **Low Average Impact:** Most critical weights have <1% individual impact
- **Embedding Dominance:** Highest sensitivities concentrated in embedding layer

## Performance Metrics Summary

### Processing Efficiency
- **Total Runtime:** ~16 minutes per complete analysis
- **Sensitivity Computation:** 6-9 seconds
- **Weight Ranking:** 34 seconds
- **Visualization Generation:** ~13 minutes
- **File I/O:** ~2.5 minutes

### Computational Scale
- **Weights per Second:** ~1.3 billion weights/second (sensitivity analysis)
- **Parameters per Second:** 142-147 parameters/second
- **Layers per Second:** 6.6 layers/second (ranking)
- **Memory Usage:** Efficient single-GPU processing

## Reproducibility Validation ✅
- **Identical Results:** P1 and P2 runs produced identical outputs
- **Consistent Timing:** Performance metrics stable across runs
- **Perfect Correlation:** Sensitivity scores match to 6+ decimal places
- **Stable Rankings:** Critical weight ordering unchanged

---
**Analysis Completed:** September 1, 2025  
**Framework:** Enhanced Sensitivity Analysis (ESA)  
**Model:** meta-llama/Llama-3.1-8B (8.03B parameters)
