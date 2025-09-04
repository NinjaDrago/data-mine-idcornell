# CSCI365_DataMining
# Drone Data Mining Project

## Overview
This project is inspired by the [colorado_river](https://github.com/wmacevoy/data-mine-wmacevoy/tree/main/colorado_river) example.  
Instead of river data, I am focusing on **drone usage**. The data comes from the City of Bloomington’s UAV flight logs, which are available through their open data portal.

The purpose of this project is to demonstrate how to **automatically collect and analyze** drone-related data as part of a data mining workflow.

---

## Repository Structure
- `get_data.py` — downloads UAV flight log data from the Bloomington open data portal and saves it locally (`data.csv`).
- `analyze.py` — performs simple analysis on the data (e.g., how many flights, when they occurred).
- `.gitignore` — ensures no actual data files are committed to the repository.
- `README.md` — project description and instructions.

---

## How to Use

### 1. Setup
Make sure you have Python 3 installed, along with `requests` and `pandas`:

```bash
pip install requests pandas