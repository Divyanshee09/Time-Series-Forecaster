# Cleanup Guide - What to Keep vs Delete

## ✅ KEEP - TMLR Submission Files (Updated & Required)

### Core TMLR Package
```
✓ TMLR_SUBMISSION_FINAL.md          ← Main paper (NEW)
✓ TMLR_VALIDATION_REPORT.md         ← Quality assurance (NEW)
✓ EXECUTIVE_SUMMARY.md              ← Overview (NEW)
✓ TRANSFORMATION_COMPLETE.md        ← Transformation log (NEW)
✓ QUICK_START.md                    ← Quick guide (NEW)
✓ benchmark_tmlr_real.csv           ← REAL results (NEW)
✓ benchmark_tmlr_run.log            ← Latest run log (NEW)
```

### Real Data & Figures (NEW)
```
✓ data/real/                        ← 5 real datasets (NEW)
  ├── m4_hourly.csv
  ├── bitcoin.csv
  ├── airline.csv
  ├── electricity.csv
  └── covid_cases.csv

✓ figures_tmlr_real/                ← 8 figures, 300 DPI (NEW)
  ├── summary_comparison.png
  ├── statistical_significance.png
  ├── diagnostic_validation.png     ← NOVEL CONTRIBUTION
  └── [5 dataset analyses]
```

### Reproducibility (NEW)
```
✓ future_scope_fixed.py             ← Source code (UPDATED)
✓ benchmark_tmlr_final.py           ← TMLR benchmark (NEW)
✓ download_real_data.py             ← Data pipeline (NEW)
✓ fix_electricity.py                ← Data fix script (NEW)
✓ Dockerfile                        ← Docker (NEW)
✓ Makefile                          ← Automation (NEW)
✓ requirements.txt                  ← Dependencies
```

---

## ❌ DELETE - Old/Obsolete Files

### Old Benchmarks (Synthetic Data - NOT for TMLR)
```
✗ benchmark_final_results.csv       ← OLD synthetic results
✗ benchmark_final_run.log           ← OLD log
✗ benchmark_optimized.log           ← Killed process log
✗ benchmark.log                     ← Very old log
✗ benchmark_direct.log              ← Test run
✗ benchmark_final.log               ← Empty/incomplete
```

### Old Data Directories (Synthetic)
```
✗ data_benchmark/                   ← OLD synthetic data
✗ data_benchmark_final/             ← OLD synthetic data
```

### Old Figure Directories (Obsolete)
```
✗ figures/                          ← OLD figures (synthetic data)
✗ figures_final/                    ← OLD figures (synthetic data)
✗ figures_quick/                    ← Test figures
```

### Old Scripts (Superseded by TMLR versions)
```
✗ benchmark_suite.py                ← Original slow version
✗ benchmark_optimized.py            ← Intermediate version
✗ benchmark_final.py                ← Superseded by benchmark_tmlr_final.py
✗ benchmark_quick.py                ← Test script
```

### Old Documentation (Superseded)
```
✗ TMLR_SUBMISSION_REPORT.md         ← v1.0 (before real data)
✗ tmlr_summary.md                   ← Old placeholder version
✗ PROJECT_STATUS.md                 ← Pre-TMLR status
✗ DELIVERABLES.md                   ← Old v1.0 checklist
```

---

## 📋 Cleanup Commands

### Safe Cleanup (Recommended)
```bash
# Create backup first
mkdir -p ../backup_old_files
mv data_benchmark data_benchmark_final ../backup_old_files/
mv figures figures_final figures_quick ../backup_old_files/
mv benchmark_suite.py benchmark_optimized.py benchmark_final.py benchmark_quick.py ../backup_old_files/
mv benchmark_final_results.csv benchmark_optimized.log benchmark.log ../backup_old_files/
mv TMLR_SUBMISSION_REPORT.md PROJECT_STATUS.md DELIVERABLES.md ../backup_old_files/

echo "✓ Old files backed up to ../backup_old_files/"
```

### Aggressive Cleanup (Delete Everything Old)
```bash
# Delete old benchmarks & logs
rm -f benchmark_final_results.csv benchmark_optimized.log benchmark.log
rm -f benchmark_direct.log benchmark_final.log

# Delete old data directories
rm -rf data_benchmark data_benchmark_final

# Delete old figure directories  
rm -rf figures figures_final figures_quick

# Delete old scripts
rm -f benchmark_suite.py benchmark_optimized.py benchmark_final.py benchmark_quick.py

# Delete old docs
rm -f TMLR_SUBMISSION_REPORT.md PROJECT_STATUS.md DELIVERABLES.md
rm -f tmlr_summary.md README_RESEARCH.md

echo "✓ Cleanup complete! Only TMLR v2.0 files remain."
```

---

## 🎯 What You'll Have After Cleanup

### TMLR Submission Directory (Clean)
```
future_scope/
├── TMLR_SUBMISSION_FINAL.md        ← Main paper
├── TMLR_VALIDATION_REPORT.md       ← QA report
├── EXECUTIVE_SUMMARY.md            ← Overview
├── TRANSFORMATION_COMPLETE.md      ← Change log
├── QUICK_START.md                  ← Guide
├── benchmark_tmlr_real.csv         ← Results
├── benchmark_tmlr_run.log          ← Log
├── data/real/                      ← 5 datasets
├── figures_tmlr_real/              ← 8 figures
├── future_scope_fixed.py           ← Code
├── benchmark_tmlr_final.py         ← Benchmark
├── download_real_data.py           ← Data script
├── fix_electricity.py              ← Data fix
├── Dockerfile                      ← Docker
├── Makefile                        ← Automation
└── requirements.txt                ← Deps

Total: ~15 files + 2 directories (vs 40+ currently)
```

---

## ⚠️ Verification Before Cleanup

Run this to verify TMLR files are complete:
```bash
# Check all required files exist
for file in TMLR_SUBMISSION_FINAL.md benchmark_tmlr_real.csv \
            future_scope_fixed.py benchmark_tmlr_final.py \
            Dockerfile Makefile requirements.txt; do
    [ -f "$file" ] && echo "✓ $file" || echo "✗ MISSING: $file"
done

# Check directories
for dir in data/real figures_tmlr_real; do
    [ -d "$dir" ] && echo "✓ $dir/" || echo "✗ MISSING: $dir/"
done

# Count figures
echo "Figures: $(ls figures_tmlr_real/*.png 2>/dev/null | wc -l) (expect 8)"

# Count datasets
echo "Datasets: $(ls data/real/*.csv 2>/dev/null | wc -l) (expect 5)"
```

If all checks pass → safe to delete old files!

---

## 🚀 Recommended Action

**Option 1 (Safe)**: Move to backup folder
```bash
cd /home/swastik/Time-Series-Forecaster/future_scope
bash -c "$(cat CLEANUP_GUIDE.md | grep -A 20 'Safe Cleanup' | tail -n +2 | head -n 8)"
```

**Option 2 (Clean Slate)**: Delete old files
```bash
cd /home/swastik/Time-Series-Forecaster/future_scope
bash -c "$(cat CLEANUP_GUIDE.md | grep -A 30 'Aggressive Cleanup' | tail -n +2 | head -n 18)"
```

**Recommendation**: Use Option 1 (Safe) - you can delete ../backup_old_files/ later after confirming TMLR submission works.
