# 04 Optimization

Performance tuning and GPU optimization for RTX 3090.

## Essential Reading
- **RTX 3090 Critical Analysis**: [`RTX3090_CRITICAL_ANALYSIS.md`](RTX3090_CRITICAL_ANALYSIS.md)
- **Reference Card**: [`RTX3090_REFERENCE_CARD.md`](RTX3090_REFERENCE_CARD.md) - Quick reference

## Deep Dives
- `RTX3090_OPTIMIZATION_GUIDE.md` - Complete optimization guide
- `WHY_ADVICE_WAS_WRONG.md` - Critical analysis of common advice
- `ML_ADVICE_VS_RL_REALITY.md` - ML vs RL differences
- `GPU_UTILIZATION_ANALYSIS.md` - Utilization breakdown

## Key Findings
- **Batch Size**: 1024 (increased from 512)
- **N Steps**: 4096 (increased from 2048)
- **cuDNN Benchmark**: Enabled
- **Expected GPU**: 10-20% for RL (not 80%+ like supervised learning)
