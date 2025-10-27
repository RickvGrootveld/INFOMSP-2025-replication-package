This File explains the cmd input to generate the experiments and files from this research project
Firstly open the cmd on your designated computer. in the cmd navigated to the project file by using:
cd [path]
Now you can use the input as seen below


GPT input zero:
python UUGPTAPI.py --mode zero "[path]\SumTree_no_comments.py" > "[path]\SumTree_commented_gpt_zero2.py"

GPT Input few
python UUGPTAPI.py --mode few "[path]SumTree_no_comments.py" ^
  "[path]\psqlextra-manager-manager.py" "[path]\Qiskit-extensions-initializer.py" ^
  "[path]\qiskit-visualization-bloch.py" ^
  > "[path]\SumTree_commented_Few_GPT.py"

Claude Input zero
python UUClaude.py --mode zero "[path]\SumTree_no_comments.py" --out "[path]\SumTree_Claude_zero.py"

Claude Input few
python UUClaude.py --mode few ^
  "[path]\SumTree_no_comments.py" ^
  "[path]\psqlextra-manager-manager.py" ^
  "[path]\Qiskit-extensions-initializer.py" ^
  "[path]\qiskit-visualization-bloch.py" ^
  --out "[path]\SumTree_Claude_few.py"

Qualtrics analisis input
Make sure that for this analisis the python file and the other csv files are in the same folder otherwise it will not work
python DataAnalisis.py ^
  --inputs "Software Production Experiment_October 16, 2025_13.26.csv" ^
           "Software Production Experiment_October 16, 2025_13.41.csv" ^
           "V2 SP Experiment_October 16, 2025_13.38.csv" ^
           "V2 SP Experiment_October 16, 2025_14.38.csv" ^
  --outdir results_paper