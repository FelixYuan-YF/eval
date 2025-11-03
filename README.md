1. Clone this repository:

   ```bash
   git clone --recursive https://github.com/FelixYuan-YF/eval.git  
   cd eval  
   ```
2. Install dependencies:

   ```bash
   conda create -n eval python=3.10 -y
   conda activate eval
   pip install -r requirements.txt
   pip install torch_scatter==2.1.2
   cd base
   python setup.py install
   ```
3. Download checkpoint files:

   ```bash
   bash download_checkpoints.sh
   ```

