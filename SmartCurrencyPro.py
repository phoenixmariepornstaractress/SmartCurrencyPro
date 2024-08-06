import tkinter as tk
from tkinter import ttk, messagebox
from forex_python.converter import CurrencyRates, CurrencyCodes
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkcalendar import DateEntry
import json
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import requests
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd

class CurrencyConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Currency Converter App")
        self.root.geometry("700x1000")
        self.currency_rates = CurrencyRates()
        self.currency_codes = CurrencyCodes()
        self.history = []
        self.favorites = []
        self.history_file = "conversion_history.json"

        # Load conversion history
        self.load_history()

        # UI elements
        self.create_widgets()

    def create_widgets(self):
        # Title
        title_label = ttk.Label(self.root, text="Currency Converter", font=("Helvetica", 16))
        title_label.pack(pady=10)

        # From currency
        from_label = ttk.Label(self.root, text="From:")
        from_label.pack(pady=5)
        self.from_currency = ttk.Combobox(self.root, values=list(self.currency_rates.get_rates("").keys()))
        self.from_currency.pack(pady=5)
        self.create_tooltip(self.from_currency, "Select the base currency")

        # Swap button
        swap_button = ttk.Button(self.root, text="Swap", command=self.swap_currencies)
        swap_button.pack(pady=5)

        # To currency
        to_label = ttk.Label(self.root, text="To:")
        to_label.pack(pady=5)
        self.to_currency = ttk.Combobox(self.root, values=list(self.currency_rates.get_rates("").keys()))
        self.to_currency.pack(pady=5)
        self.create_tooltip(self.to_currency, "Select the target currency")

        # Amount
        amount_label = ttk.Label(self.root, text="Amount:")
        amount_label.pack(pady=5)
        self.amount_entry = ttk.Entry(self.root)
        self.amount_entry.pack(pady=5)
        self.create_tooltip(self.amount_entry, "Enter the amount to convert")

        # Convert button
        convert_button = ttk.Button(self.root, text="Convert", command=self.convert_currency)
        convert_button.pack(pady=10)

        # Clear button
        clear_button = ttk.Button(self.root, text="Clear", command=self.clear_fields)
        clear_button.pack(pady=10)

        # Favorite button
        favorite_button = ttk.Button(self.root, text="Add to Favorites", command=self.add_to_favorites)
        favorite_button.pack(pady=10)

        # Result
        self.result_label = ttk.Label(self.root, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=20)

        # History
        history_label = ttk.Label(self.root, text="Conversion History", font=("Helvetica", 14))
        history_label.pack(pady=10)
        self.history_text = tk.Text(self.root, height=10, width=70)
        self.history_text.pack(pady=10)
        self.update_history()

        # Favorite Currencies
        favorite_label = ttk.Label(self.root, text="Favorite Currencies", font=("Helvetica", 14))
        favorite_label.pack(pady=10)
        self.favorite_text = tk.Text(self.root, height=5, width=70)
        self.favorite_text.pack(pady=10)
        self.update_favorites()

        # Favorite Currencies Dropdown
        self.favorite_currency_var = tk.StringVar()
        self.favorite_currency_dropdown = ttk.Combobox(self.root, textvariable=self.favorite_currency_var, values=self.favorites)
        self.favorite_currency_dropdown.pack(pady=10)
        self.create_tooltip(self.favorite_currency_dropdown, "Select a favorite currency pair")

        # Load favorite button
        load_favorite_button = ttk.Button(self.root, text="Load Favorite", command=self.load_favorite_currency)
        load_favorite_button.pack(pady=10)

        # Live Exchange Rates
        live_label = ttk.Label(self.root, text="Live Exchange Rate", font=("Helvetica", 14))
        live_label.pack(pady=10)
        self.live_rate_label = ttk.Label(self.root, text="", font=("Helvetica", 12))
        self.live_rate_label.pack(pady=10)

        # Historical Graph
        graph_button = ttk.Button(self.root, text="Show Historical Rates", command=self.show_graph)
        graph_button.pack(pady=10)

        # Date Range for Historical Graph
        date_label = ttk.Label(self.root, text="Select Date Range", font=("Helvetica", 14))
        date_label.pack(pady=10)

        start_date_label = ttk.Label(self.root, text="Start Date:")
        start_date_label.pack(pady=5)
        self.start_date_entry = DateEntry(self.root)
        self.start_date_entry.pack(pady=5)
        self.create_tooltip(self.start_date_entry, "Select the start date for historical rates")

        end_date_label = ttk.Label(self.root, text="End Date:")
        end_date_label.pack(pady=5)
        self.end_date_entry = DateEntry(self.root)
        self.end_date_entry.pack(pady=5)
        self.create_tooltip(self.end_date_entry, "Select the end date for historical rates")

        # Predict Button
        predict_button = ttk.Button(self.root, text="Predict Future Rates", command=self.predict_rates)
        predict_button.pack(pady=10)

        # Prediction Result
        self.prediction_label = ttk.Label(self.root, text="", font=("Helvetica", 12))
        self.prediction_label.pack(pady=10)

        # Sentiment Analysis Button
        sentiment_button = ttk.Button(self.root, text="Analyze Sentiment of Currency News", command=self.analyze_sentiment)
        sentiment_button.pack(pady=10)

        # Sentiment Result
        self.sentiment_label = ttk.Label(self.root, text="", font=("Helvetica", 12))
        self.sentiment_label.pack(pady=10)

        # Anomaly Detection Button
        anomaly_button = ttk.Button(self.root, text="Detect Anomalies in Rates", command=self.detect_anomalies)
        anomaly_button.pack(pady=10)

        # Anomaly Detection Result
        self.anomaly_label = ttk.Label(self.root, text="", font=("Helvetica", 12))
        self.anomaly_label.pack(pady=10)

    def convert_currency(self):
        try:
            from_curr = self.from_currency.get()
            to_curr = self.to_currency.get()
            amount = float(self.amount_entry.get())
            converted_amount = self.currency_rates.convert(from_curr, to_curr, amount)
            result_text = f"{amount} {self.currency_codes.get_symbol(from_curr)} ({self.currency_codes.get_currency_name(from_curr)}) = {converted_amount:.2f} {self.currency_codes.get_symbol(to_curr)} ({self.currency_codes.get_currency_name(to_curr)})"
            self.result_label.config(text=result_text)

            # Add to history
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.history.append(f"{timestamp} - {result_text}")
            self.update_history()

            # Update live exchange rate
            rate = self.currency_rates.get_rate(from_curr, to_curr)
            live_rate_text = f"1 {from_curr} = {rate:.4f} {to_curr}"
            self.live_rate_label.config(text=live_rate_text)

            # Save history to file
            self.save_history()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def clear_fields(self):
        self.from_currency.set("")
        self.to_currency.set("")
        self.amount_entry.delete(0, tk.END)
        self.result_label.config(text="")

    def swap_currencies(self):
        from_curr = self.from_currency.get()
        to_curr = self.to_currency.get()
        self.from_currency.set(to_curr)
        self.to_currency.set(from_curr)

    def add_to_favorites(self):
        from_curr = self.from_currency.get()
        to_curr = self.to_currency.get()
        if from_curr and to_curr:
            favorite_pair = f"{from_curr} -> {to_curr}"
            if favorite_pair not in self.favorites:
                self.favorites.append(favorite_pair)
                self.update_favorites()

    def update_history(self):
        self.history_text.delete(1.0, tk.END)
        for record in self.history[-10:]:
            self.history_text.insert(tk.END, record + "\n")

    def update_favorites(self):
        self.favorite_text.delete(1.0, tk.END)
        for pair in self.favorites:
            self.favorite_text.insert(tk.END, pair + "\n")
        self.favorite_currency_dropdown.config(values=self.favorites)

    def load_favorite_currency(self):
        selected_pair = self.favorite_currency_var.get()
        if selected_pair:
            from_curr, to_curr = selected_pair.split(" -> ")
            self.from_currency.set(from_curr)
            self.to_currency.set(to_curr)

    def save_history(self):
        with open(self.history_file, "w") as f:
            json.dump(self.history, f)

    def load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                self.history = json.load(f)

    def show_graph(self):
        try:
            from_curr = self.from_currency.get()
            to_curr = self.to_currency.get()
            start_date = self.start_date_entry.get_date()
            end_date = self.end_date_entry.get_date()

            dates = []
            rates = []
            current_date = start_date

            while current_date <= end_date:
                rate = self.currency_rates.get_rate(from_curr, to_curr, current_date)
                dates.append(current_date)
                rates.append(rate)
                current_date += datetime.timedelta(days=1)

            plt.figure(figsize=(10, 6))
            plt.plot(dates, rates, marker='o')
            plt.title(f"Historical Exchange Rates: {from_curr} to {to_curr}")
            plt.xlabel("Date")
            plt.ylabel("Exchange Rate")
            plt.grid(True)
            plt.xticks(rotation=45)

            fig = plt.gcf()
            canvas = FigureCanvasTkAgg(fig, master=self.root)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=20)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def predict_rates(self):
        try:
            from_curr = self.from_currency.get()
            to_curr = self.to_currency.get()
            start_date = self.start_date_entry.get_date()
            end_date = self.end_date_entry.get_date()

            dates = []
            rates = []
            current_date = start_date

            while current_date <= end_date:
                rate = self.currency_rates.get_rate(from_curr, to_curr, current_date)
                dates.append(current_date)
                rates.append(rate)
                current_date += datetime.timedelta(days=1)

            df = pd.DataFrame({'date': dates, 'rate': rates})
            df['timestamp'] = df['date'].map(pd.Timestamp.timestamp)
            X = np.array(df['timestamp']).reshape(-1, 1)
            y = np.array(df['rate'])

            model = LinearRegression()
            model.fit(X, y)

            future_dates = pd.date_range(start=end_date, periods=30).tolist()
            future_timestamps = np.array([pd.Timestamp(date).timestamp() for date in future_dates]).reshape(-1, 1)
            future_rates = model.predict(future_timestamps)

            plt.figure(figsize=(10, 6))
            plt.plot(dates, rates, marker='o', label="Historical Rates")
            plt.plot(future_dates, future_rates, marker='x', linestyle='dashed', label="Predicted Rates")
            plt.title(f"Historical and Predicted Exchange Rates: {from_curr} to {to_curr}")
            plt.xlabel("Date")
            plt.ylabel("Exchange Rate")
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)

            fig = plt.gcf()
            canvas = FigureCanvasTkAgg(fig, master=self.root)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=20)

            prediction_text = f"Predicted rate on {future_dates[-1].date()}: {future_rates[-1]:.4f} {to_curr}"
            self.prediction_label.config(text=prediction_text)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def analyze_sentiment(self):
        try:
            from_curr = self.from_currency.get()
            to_curr = self.to_currency.get()
            news_api_key = "YOUR_NEWSAPI_KEY"
            url = f"https://newsapi.org/v2/everything?q={from_curr}+{to_curr}&apiKey={news_api_key}"
            response = requests.get(url)
            news_data = response.json()
            articles = news_data['articles']

            sentiments = []
            for article in articles:
                description = article['description']
                if description:
                    analysis = TextBlob(description)
                    sentiments.append(analysis.sentiment.polarity)

            if sentiments:
                avg_sentiment = np.mean(sentiments)
                sentiment_text = f"Average Sentiment Polarity for {from_curr}/{to_curr} news: {avg_sentiment:.2f}"
                self.sentiment_label.config(text=sentiment_text)
            else:
                self.sentiment_label.config(text="No relevant news articles found.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def detect_anomalies(self):
        try:
            from_curr = self.from_currency.get()
            to_curr = self.to_currency.get()
            start_date = self.start_date_entry.get_date()
            end_date = self.end_date_entry.get_date()

            dates = []
            rates = []
            current_date = start_date

            while current_date <= end_date:
                rate = self.currency_rates.get_rate(from_curr, to_curr, current_date)
                dates.append(current_date)
                rates.append(rate)
                current_date += datetime.timedelta(days=1)

            rates_np = np.array(rates)
            mean = np.mean(rates_np)
            std_dev = np.std(rates_np)
            anomalies = [(date, rate) for date, rate in zip(dates, rates) if abs(rate - mean) > 2 * std_dev]

            anomaly_text = "Anomalies detected:\n" + "\n".join([f"{date}: {rate}" for date, rate in anomalies])
            self.anomaly_label.config(text=anomaly_text)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def create_tooltip(self, widget, text):
        tooltip = tk.Label(self.root, text=text, bg="yellow", bd=1, relief=tk.SOLID)
        tooltip.pack_forget()

        def show_tooltip(event):
            tooltip.place(x=widget.winfo_rootx(), y=widget.winfo_rooty() - 20)
            tooltip.pack()

        def hide_tooltip(event):
            tooltip.pack_forget()

        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)

if __name__ == "__main__":
    root = tk.Tk()
    app = CurrencyConverterApp(root)
    root.mainloop()
