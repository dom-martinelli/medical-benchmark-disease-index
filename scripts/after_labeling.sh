#!/bin/zsh
# Waits for Stage 2 labeling to finish, then re-runs the later stages in fresh
# processes so they use the current code (the labeling process loaded stages/code.py
# before the version 5 lookup existed). Runs one step at a time to spare memory.
set -u
cd "/Users/dominiero/Documents/ChatGPT/sep benchmark paper/benchmark-index"
LOG=data/after_labeling.log
S=/private/tmp/claude-501/-Users-dominiero/2b085120-a064-40bc-8c3d-48e9b38d1a87/scratchpad
echo "$(date) waiting for Stage 2 to complete" >> $LOG
until grep -q "Stage 2 complete" run_log.md; do
  if ! pgrep -f "run.py --approve-pilot" >/dev/null; then
    echo "$(date) labeling process exited without logging Stage 2 complete; stopping here" >> $LOG; exit 1
  fi
  sleep 60
done
echo "$(date) Stage 2 complete; stopping the labeling process before it codes with the old lookup" >> $LOG
pkill -f "run.py --approve-pilot"; sleep 15
for stage in code validate report; do
  echo "$(date) starting stage $stage" >> $LOG
  caffeinate -i .venv/bin/python run.py --stage $stage >> $LOG 2>&1 || { echo "$(date) stage $stage failed" >> $LOG; exit 1; }
done
echo "$(date) external MedMCQA subject check" >> $LOG
caffeinate -i .venv/bin/python -c "from stages import external_check; external_check.run()" >> $LOG 2>&1
echo "$(date) lookup before/after measurement" >> $LOG
nice -n 15 .venv/bin/python $S/measure.py > $S/measure_v5_out.txt 2>&1
echo "$(date) all done" >> $LOG
