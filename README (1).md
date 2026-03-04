# Data

## UCI Bank Marketing Dataset

This project uses the **Bank Marketing** dataset from the UCI Machine Learning Repository.

### Download Instructions

1. Visit the dataset page:  
   👉 https://archive.ics.uci.edu/dataset/222/bank+marketing

2. Download `bank-additional.zip`

3. Extract and place `bank-additional-full.csv` in this `data/` folder.

4. Run the pipeline with:
   ```bash
   python main.py --data data/bank-additional-full.csv
   ```

### Citation

> Moro, S., Cortez, P., & Rita, P. (2014).  
> *A data-driven approach to predict the success of bank telemarketing.*  
> Decision Support Systems, 62, 22-31.  
> DOI: 10.1016/j.dss.2014.03.001

### Dataset Summary

| Property        | Value                          |
|-----------------|-------------------------------|
| Instances       | 45,211 (full) / 41,188 (extra)|
| Features        | 20 + 1 target                 |
| Target classes  | yes / no                      |
| Positive rate   | ~11.3%                        |
| Format          | Semicolon-separated CSV        |

> **Note:** The `data/` folder is listed in `.gitignore` to avoid committing large data files to version control.
