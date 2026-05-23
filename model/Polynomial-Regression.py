import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sklearn.model_selection import train_test_split

matplotlib.use("TkAgg")

# MODEL 
class PolynomialRegression:
    def __init__(self, degree, lr=1e-1, epoch=10000):
        self.lr = lr
        self.epoch = epoch
        self.degree = degree

    def fit(self, x, yt):
        if yt.ndim == 1:
            yt = yt.reshape(-1, 1)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        np.random.seed(42)
        X = [(x**i) for i in range(1, self.degree + 1)]
        X = np.concatenate(X, axis=1)

        m, n = X.shape
        self.w = np.random.randn(n, 1)
        self.b = np.random.randn(1,)

        self.mu = X.mean(axis=0)
        self.std = X.std(axis=0)
        X = (X - self.mu) / self.std

        prev_loss = float('inf')

        for i in range(self.epoch):
            yp = (X @ self.w + self.b).reshape(-1, 1)
            loss = 0.5 * ((yp - yt) ** 2).mean()

            grad = (X.T @ (yp - yt)) / m
            self.w -= self.lr * grad
            self.b -= self.lr * (yp - yt).mean()

            if abs(prev_loss - loss) < 1e-6:
                break
            prev_loss = loss

    def predict(self, x):
        X1 = [(x**i) for i in range(1, self.degree + 1)]
        X1 = np.concatenate(X1, axis=1)
        X1 = (X1 - self.mu) / self.std
        return X1 @ self.w + self.b


def r2_score(yt, yp):
    sse = ((yt - yp) ** 2).sum()
    sst = ((yt - yt.mean()) ** 2).sum()
    return 1 - (sse / sst)


def best_fit(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    x_train, x_test = x_train.to_numpy(), x_test.to_numpy()
    y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

    y_test = y_test.reshape(-1, 1)

    models = [PolynomialRegression(degree=i) for i in range(1, 5)]

    for model in models:
        model.fit(x_train, y_train)

    yps = np.array([model.predict(x_test).ravel() for model in models]).T

    sse = ((y_test - yps) ** 2).sum(axis=0)
    sst = ((y_test - y_test.mean()) ** 2).sum()

    score = 1 - (sse / sst)
    return np.argmax(score)


def build_regression_model(degree):
    return PolynomialRegression(degree)


def preprocess(df, x_col, y_col):
    y = df[y_col]
    x = df[x_col]

    cat_cols = x.select_dtypes(include=['object']).columns
    num_cols = x.select_dtypes(include=['int64', 'float64']).columns

    x = pd.get_dummies(x, columns=cat_cols, dtype=int)
    x[num_cols] = (x[num_cols] - x[num_cols].mean()) / x[num_cols].std()

    return x, y


# GUI 
class RegressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Regression Analyzer")
        self.root.geometry("1100x750")

        self.dataframe = None
        self.numeric_columns = []
        self.current_file = None

        self.file_var = tk.StringVar(value="No CSV selected")
        self.shape_var = tk.StringVar(value="X shape: -    Y shape: -")
        self.metrics_var = tk.StringVar(value="Load CSV to begin.")
        self.status_var = tk.StringVar(value="Auto detect enabled")

        self._build_layout()

    def _build_layout(self):
        outer = ttk.Frame(self.root, padding=15)
        outer.pack(fill="both", expand=True)

        # ===== TOP =====
        top = ttk.LabelFrame(outer, text="CSV Input", padding=10)
        top.pack(fill="x", pady=10)

        ttk.Button(top, text="Open CSV", command=self.open_csv)\
            .grid(row=0, column=0, padx=5, pady=5)

        ttk.Label(top, textvariable=self.file_var)\
            .grid(row=0, column=1, columnspan=3, sticky="w")

        ttk.Label(top, text="Target (Y):")\
            .grid(row=1, column=0, padx=5)

        self.target_combo = ttk.Combobox(top, state="readonly", width=25)
        self.target_combo.grid(row=1, column=1)

        ttk.Button(top, text="Run Analysis", command=self.run_analysis)\
            .grid(row=1, column=2, padx=10)

        ttk.Label(top, textvariable=self.shape_var)\
            .grid(row=2, column=0, columnspan=4, sticky="w")

        ttk.Label(top, textvariable=self.status_var, foreground="gray")\
            .grid(row=3, column=0, columnspan=4, sticky="w")

        # ===== MIDDLE =====
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True)

        left = ttk.LabelFrame(middle, text="Preview", padding=10)
        left.pack(side="left", fill="both", padx=5)

        

        right = ttk.LabelFrame(middle, text="Results", padding=10)
        right.pack(side="left", fill="both", expand=True)

        self.metrics_label = ttk.Label(right, textvariable=self.metrics_var, justify="left")
        self.metrics_label.pack(anchor="w", pady=5)

        self.figure = Figure(figsize=(8, 5))
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)

        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    
    def open_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return

        df = pd.read_csv(path)

        self.dataframe = df
        self.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        self.file_var.set(path)
        self.target_combo["values"] = self.numeric_columns
        self.target_combo.set(self.numeric_columns[-1])
        self.run_analysis()

        

    def run_analysis(self):
        if self.dataframe is None:
            return

        y_col = self.target_combo.get()
        x_cols = [c for c in self.numeric_columns if c != y_col]

        df = self.dataframe[x_cols + [y_col]].dropna()

        X, y = preprocess(df, x_cols, y_col)

        score = best_fit(X, y)

        model = build_regression_model(score + 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X.to_numpy(), y.to_numpy(), test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        y_test = y_test.reshape(-1, 1)

        mse = ((y_test - preds) ** 2).mean()
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - preds))
        r2 = r2_score(y_test, preds)
        
        if score==0:
            scre='Linear Regression'
        else:
            scre=f'Polynomial Regression of Degree {score+1}'
        self.metrics_var.set(
            f"Best Model: {scre}\n"
            f"R2: {r2:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}"
        )

        self.ax1.clear()
        self.ax2.clear()
        min_val = min(float(np.min(y_test)), np.min(preds))
        max_val = max(np.max(y_test), np.max(preds))

        self.ax1.scatter(y_test, preds)
        self.ax1.plot([min_val,max_val],[min_val,max_val],'r--',color='red')
        
        self.ax1.set_title("Actual vs Predicted")

        residuals = y_test - preds
        self.ax2.scatter(preds, residuals)
        self.ax2.axhline(0, color="red", linestyle="--", linewidth=1.5)
        #self.ax2.axhline(0)

        self.canvas.draw()


def main():
    root = tk.Tk()
    RegressionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()