import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to the SQLite database
conn = sqlite3.connect('log.db')
QUERY = """select epoch, train_loss, test_loss from LogConfig where run_id = 226041554964575;"""

data = pd.read_sql(QUERY, conn)
print(data)

plt.plot(data["epoch"], data["train_loss"], label="train_loss")
plt.plot(data["epoch"], data["test_loss"], label="test_loss")
plt.grid()
plt.legend()

plt.savefig("/eos/home-d/davit/train_test.png")
