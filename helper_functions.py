from sklearn.metrics import confusion_matrix
import sqlite3
from sqlalchemy import create_engine
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def create_connection(db_file):
    """
    Create a DataBase connection to the SQLite DB specified by db_file.
    Args:
        db_file: DataBase File.
    Returns:
        Connection object or None.
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Exception as e:
        print(e)
    return None


def check_table_exists(db_connection):
    """
    Function to get all the Tables in the DataBase.
    Args:
        db_connection: DataBase connection object.
    Returns:
        int - Number of tables in the DataBase.
    """
    cur = db_connection.cursor()
    query_string = "Select name from sqlite_master where type='table'"
    table_names = cur.execute(query_string)
    print(f"Tables in the DB:")
    tables = table_names.fetchall()
    print(tables[0][0])
    return len(tables)


def plot_conf_matrix(y_true, y_preds):
    """
    Function to plot Confusion Matrix.
    Args:
        - y_true: Actual y values (target labels).
        - y_preds: Predicted y values.
    """
    conf_matrix = confusion_matrix(y_true, y_preds)
    ax = sns.heatmap(conf_matrix, annot=True, fmt='g')
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix', fontsize=16)
    plt.show()
