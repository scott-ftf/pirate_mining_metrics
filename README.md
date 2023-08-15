This script interacts with the Pirate Chain daemon RPC to scrape statistical data from a set of blocks to save in a parquet database file for running statitical annlytics on. At the end, it produces a pdf report.

A example report can be found here:

A example parquet data file can be found here: 

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

