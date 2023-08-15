This script interacts with the Pirate Chain daemon RPC to extract statistical data from a set of blocks. This data is then saved in a Parquet database file for subsequent statistical analysis. At the end, the script generates a PDF report.

A example report can be found here: <br />
https://github.com/scott-ftf/pirate_mining_metrics/blob/main/pirate_miner_metrics_report.pdf

A example parquet data file can be found here: <br />
https://github.com/scott-ftf/pirate_mining_metrics/blob/main/block_data_2420000_to_2523958.parquet

---

Requires python3 and a running Pirate Chain Daemon

Required libraries:
```BASH
pip install pandas matplotlib seaborn scikit-learn reportlab Pillow
```

---

To run, start Pirate Chain daemon, and update the configuration section of `process_chain_metrics.py` with the RPC username, password, and host:port

Run with 
```BASH
python3 process_chain_metrics.py
```
Once complete, a parquet data file will be produced.

Next, run the report generator:
```BASH
python3 create_report.py <filename.parquet>
```
Once complete a .pdf report will be generated

