default: F7 F4 

J=jupyter nbconvert  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0 --allow-errors --execute
JN=$(J) --to notebook --inplace

A1:
	$(JN) RESULTS_03_NMNIST.ipynb
	git commit -m'results : RESULTS_03 ' RESULTS_03_NMNIST.ipynb
	git push
A2:
	$(JN) RESULTS_02_PokerDVS.ipynb
	git commit -m'results : RESULTS_02 ' RESULTS_02_PokerDVS.ipynb
	git push
A3:
	$(JN) RESULTS_01_SimpleAlphabet.ipynb
	git commit -m'results : RESULTS_01 ' RESULTS_01_SimpleAlphabet.ipynb
	git push
B:    
	$(JN) figures.ipynb
	git commit -m'results : figures ' figures.ipynb
	git push
C:    
	$(JN) BASICS_04_Hacks.ipynb
	git commit -m'results : basics ' BASICS_04_Hacks.ipynb
	git push
D:
	$(JN) HOTS_EXP_03_Pytorch_LR.ipynb
	git commit -m 'EXP 03 results' HOTS_EXP_03_Pytorch_LR.ipynb
	git push
E:
	$(JN) HOTS_EXP_05_architectures.ipynb
	git commit -m 'EXP 05 results' HOTS_EXP_05_architectures.ipynb
	git push

F: F1 F2 F3 F4 F5 F6
F1:
	python3 homhots_full.py
	python3 homhots_nmnist_Kfold.py
F2:
	python3 homhots_ncars.py
F3:
	python3 homhots_archiz.py
F4:
	python3 homhots_LR.py
F5:
	python3 homhots_nmnist_jitter.py
F6:
	python3 homhots_gesture.py
F7:
	python3 homhots_ncars_bigger.py

G:
	$(JN) HOTS_EXP_06_PokerDVS.ipynb
	git commit -m 'EXP 06 results' HOTS_EXP_06_PokerDVS.ipynb
	git push
