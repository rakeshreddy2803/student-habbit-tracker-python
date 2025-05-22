import json
import datetime
import pandas as pd
import streamlit as st
import random
import hashlib
import numpy as np
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
from nltk import download as nltk_download
from datetime import timedelta
from streamlit import toast

# Initialize NLTK data for TextBlob
try:
    nltk_download('punkt')
    nltk_download('averaged_perceptron_tagger')
except:
    pass

# Define categories for savings
CATEGORIES = ["Savings", "Food", "Clothing", "Trip Savings", "Education", "Transport", "Entertainment"]

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"
if 'form_active' not in st.session_state:
    st.session_state.form_active = False

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Data loading functions
def load_data(filename):
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# Data saving functions
def save_data(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

# Initialize data files
def init_files():
    for file in ["users.json", "wallet.json", "spending_plans.json"]:
        try:
            with open(file, "r") as f:
                json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            with open(file, "w") as f:
                json.dump({}, f)

init_files()

# AI Model Classes
class SavingsPredictor:
    def __init__(self):
        self.model = LinearRegression()
        
    def train(self, transactions):
        if len(transactions) < 7:
            return False
            
        df = pd.DataFrame(transactions)
        try:
            df['date'] = pd.to_datetime(df['date'])
            daily = df.groupby('date')['amount'].sum().reset_index()
            daily['day_num'] = (daily['date'] - daily['date'].min()).dt.days
            
            X = daily['day_num'].values.reshape(-1, 1)
            y = daily['amount'].values
            self.model.fit(X, y)
            return True
        except:
            return False
        
    def predict(self, days_ahead):
        last_day = len(self.model.coef_)
        future_days = np.array([last_day + i for i in range(1, days_ahead+1)]).reshape(-1, 1)
        return self.model.predict(future_days)

class ExpenseCategorizer:
    def __init__(self):
        self.keywords = {
            'Food': ['restaurant', 'cafe', 'groceries', 'food', 'eat', 'dinner', 'lunch'],
            'Transport': ['uber', 'lyft', 'taxi', 'bus', 'train', 'metro', 'gas', 'fuel'],
            'Entertainment': ['movie', 'netflix', 'concert', 'game', 'hobby'],
            'Education': ['book', 'course', 'tuition', 'school', 'university'],
            'Clothing': ['shirt', 'pants', 'dress', 'shoes', 'clothes']
        }
        
    def categorize(self, description):
        if not description:
            return "Other"
            
        description = description.lower()
        for category, terms in self.keywords.items():
            if any(term in description for term in terms):
                return category
        return "Other"

class AnomalyDetector:
    def __init__(self):
        self.threshold = 2.5
        
    def detect(self, transactions):
        if len(transactions) < 10:
            return []
            
        df = pd.DataFrame(transactions)
        amounts = df['amount'].abs()
        mean = amounts.mean()
        std = amounts.std()
        
        anomalies = df[(amounts > mean + self.threshold * std) | 
                      (amounts < mean - self.threshold * std)]
        return anomalies.to_dict('records')

# Main SavingTracker Class
class SavingTracker:
    def __init__(self, username):
        self.username = username
        self.wallet = load_data("wallet.json")
        self.spending_plans = load_data("spending_plans.json")
        self.predictor = SavingsPredictor()
        self.categorizer = ExpenseCategorizer()
        self.detector = AnomalyDetector()
        
        if username not in self.wallet:
            self.wallet[username] = {
                "balance": 0.0,
                "transactions": [],
                "goals": {},
                "total_amount": 0.0  # New field for total amount
            }
            save_data(self.wallet, "wallet.json")
            
        if username not in self.spending_plans:
            self.spending_plans[username] = {
                "daily": {},
                "monthly": {cat: 0.0 for cat in CATEGORIES if cat != "Savings"}
            }
            save_data(self.spending_plans, "spending_plans.json")
        else:
            # Ensure both daily and monthly keys exist for existing users
            if "daily" not in self.spending_plans[username]:
                self.spending_plans[username]["daily"] = {}
            if "monthly" not in self.spending_plans[username]:
                self.spending_plans[username]["monthly"] = {cat: 0.0 for cat in CATEGORIES if cat != "Savings"}
            save_data(self.spending_plans, "spending_plans.json")

    def refresh_data(self):
        """Reload all data from files"""
        self.wallet = load_data("wallet.json")
        self.spending_plans = load_data("spending_plans.json")

    def set_total_amount(self, amount):
        """Set the total amount available for savings and spending"""
        if amount < 0:
            st.error("Amount must be positive")
            return False
            
        self.wallet[self.username]["total_amount"] = amount
        save_data(self.wallet, "wallet.json")
        return True

    def add_saving(self, amount, category):
        if amount < 100:
            st.error("Minimum savings amount is ₹100")
            return
            
        total_available = self.wallet[self.username]["total_amount"]
        if total_available > 0 and (self.wallet[self.username]["balance"] + amount) > total_available:
            st.error(f"Cannot exceed total available amount of ₹{total_available:.2f}")
            return
            
        date = str(datetime.date.today())
        self.wallet[self.username]["balance"] += amount
        self.wallet[self.username]["transactions"].append({
            "type": "Saving", 
            "amount": amount, 
            "category": category, 
            "date": date
        })
        save_data(self.wallet, "wallet.json")
        st.success(f"Saved ₹{amount:.2f} in {category}")
        st.balloons()
        st.rerun()

    def spend_money(self, amount, category, description=""):
        if amount <= 0:
            st.error("Amount must be positive")
            return
        
        total_available = self.wallet[self.username]["total_amount"]
        if total_available > 0 and (self.wallet[self.username]["balance"] - amount) < 0:
            st.error(f"Cannot go below zero balance (total available: ₹{total_available:.2f})")
            return

        if self.wallet[self.username]["balance"] < amount:
            st.error("Insufficient funds!")
            return

        date = str(datetime.date.today())
        self.wallet[self.username]["balance"] -= amount
        self.wallet[self.username]["transactions"].append({
            "type": "Spend", 
            "amount": -amount, 
            "category": category, 
            "date": date,
            "description": description
        })
        save_data(self.wallet, "wallet.json")
        st.success(f"Spent ₹{amount:.2f} on {category}")
        st.rerun()

    def show_spending_plan(self):
        st.subheader("📝 Budget Planner")
        
        tab1, tab2, tab3 = st.tabs(["Daily Budget", "Monthly Budget", "Total Amount"])
        
        with tab1:
            date = str(datetime.date.today())
            
            # Initialize daily plan if not exists
            if date not in self.spending_plans[self.username]["daily"]:
                self.spending_plans[self.username]["daily"][date] = {
                    cat: 0.0 for cat in CATEGORIES if cat != "Savings"
                }
                save_data(self.spending_plans, "spending_plans.json")
            
            current_plan = self.spending_plans[self.username]["daily"][date]
            total_planned = sum(current_plan.values())
            remaining = self.wallet[self.username]["balance"] - total_planned
            
            cols = st.columns(2)
            cols[0].metric("Total Planned", f"₹{total_planned:.2f}")
            cols[1].metric("Remaining Balance", f"₹{remaining:.2f}", 
                         delta_color="inverse" if remaining < 0 else "normal")
            
            st.write("### Set Your Daily Budget")
            updated_plan = {}
            for category in current_plan:
                updated_plan[category] = st.number_input(
                    f"Amount for {category}",
                    min_value=0.0,
                    value=float(current_plan[category]),
                    step=10.0,
                    key=f"daily_{category}"
                )
            
            if st.button("Update Daily Plan"):
                self.spending_plans[self.username]["daily"][date] = updated_plan
                save_data(self.spending_plans, "spending_plans.json")
                st.success("Daily plan updated!")
                st.rerun()
        
        with tab2:
            st.write("### Monthly Budget Allocation")
            monthly_plan = self.spending_plans[self.username]["monthly"]
            
            updated_monthly = {}
            for category in monthly_plan:
                updated_monthly[category] = st.number_input(
                    f"Monthly limit for {category}",
                    min_value=0.0,
                    value=float(monthly_plan[category]),
                    step=100.0,
                    key=f"monthly_{category}"
                )
            
            if st.button("Update Monthly Budget"):
                self.spending_plans[self.username]["monthly"] = updated_monthly
                save_data(self.spending_plans, "spending_plans.json")
                st.success("Monthly budget updated!")
                st.rerun()
                
        with tab3:
            st.write("### Set Total Available Amount")
            current_total = self.wallet[self.username].get("total_amount", 0.0)
            new_total = st.number_input(
                "Total amount available for savings and spending",
                min_value=0.0,
                value=float(current_total),
                step=100.0
            )
            
            if st.button("Set Total Amount"):
                if self.set_total_amount(new_total):
                    st.success(f"Total amount set to ₹{new_total:.2f}")
                    st.rerun()

    def get_spending_summary(self):
        if not self.wallet[self.username]["transactions"]:
            return {"total": 0, "today": 0, "week": 0, "month": 0, "year": 0}

        df = pd.DataFrame(self.wallet[self.username]["transactions"])
        try:
            df["date"] = pd.to_datetime(df["date"])
        except ValueError:
            st.error("Invalid date format in transactions")
            return {"total": 0, "today": 0, "week": 0, "month": 0, "year": 0}

        today = pd.Timestamp(datetime.date.today())
        week_start = today - pd.Timedelta(days=7)
        month_start = today.replace(day=1)
        year_start = today.replace(month=1, day=1)

        summary = {
            "total": round(df["amount"].sum(), 2),
            "today": round(df[df["date"].dt.date == today.date()]["amount"].sum(), 2),
            "week": round(df[df["date"] >= week_start]["amount"].sum(), 2),
            "month": round(df[df["date"] >= month_start]["amount"].sum(), 2),
            "year": round(df[df["date"] >= year_start]["amount"].sum(), 2)
        }
        return summary

    def show_notifications(self):
        savings_quotes = [
            "💡 Small savings add up to big dreams!",
            "💰 Consistency is the key to financial freedom",
            "📈 Invest in your future self today"
        ]
        monthly_savings = self.get_spending_summary()["month"]
        
        with st.sidebar:
            st.subheader("🔔 Notifications")
            if monthly_savings > 0:
                st.success(f"✅ Saved ₹{monthly_savings:.2f} this month!")
            else:
                st.warning("⚠️ No savings this month yet")
            
            # Show total amount status if set
            total_amount = self.wallet[self.username].get("total_amount", 0)
            if total_amount > 0:
                balance = self.wallet[self.username]["balance"]
                remaining = total_amount - balance
                st.metric("Total Available", f"₹{total_amount:.2f}", f"₹{remaining:.2f} remaining")
            
            st.write(random.choice(savings_quotes))

    def get_ai_insights(self):
        insights = []
        transactions = self.wallet[self.username]["transactions"]
        
        if not transactions:
            return ["Start saving to get personalized insights!"]
            
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        
        # Spending analysis
        spending = df[df['amount'] < 0]
        if not spending.empty:
            top_category = spending.groupby('category')['amount'].sum().idxmin()
            insights.append(f"💸 Highest spending: {top_category}")
            
            weekly_spending = spending.resample('W', on='date')['amount'].sum().mean()
            insights.append(f"📆 Weekly average: ₹{abs(weekly_spending):.2f}")
        
        # Savings analysis
        savings = df[df['amount'] > 0]
        if not savings.empty:
            avg_saving = savings['amount'].mean()
            last_saving = savings.iloc[-1]['amount']
            if last_saving < avg_saving * 0.7:
                insights.append("⚠️ Saving less than usual")
            elif last_saving > avg_saving * 1.3:
                insights.append("🎉 Saving more than usual")
        
        # Predictions
        if self.predictor.train(transactions):
            next_week = self.predictor.predict(7)
            predicted_savings = sum(x for x in next_week if x > 0)
            insights.append(f"🔮 Next week: ₹{predicted_savings:.2f} potential savings")
        
        # Anomalies
        anomalies = self.detector.detect(transactions)
        if anomalies:
            insights.append("🚨 Unusual transactions detected")
        
        return insights if insights else ["Keep using the app for more insights"]

    def show_ai_dashboard(self):
        st.subheader("🤖 AI Insights Dashboard")
        
        tab1, tab2, tab3 = st.tabs(["Insights", "Forecast", "Anomalies"])
        
        with tab1:
            st.write("### Personalized Insights")
            insights = self.get_ai_insights()
            for insight in insights:
                st.info(insight)
            
            st.write("### Spending Patterns")
            df = pd.DataFrame(self.wallet[self.username]["transactions"])
            if not df.empty:
                spending = df[df['amount'] < 0]
                if not spending.empty:
                    by_category = spending.groupby('category')['amount'].sum().abs()
                    st.bar_chart(by_category)
        
        with tab2:
            if self.predictor.train(self.wallet[self.username]["transactions"]):
                st.write("### Savings Forecast")
                days = st.slider("Forecast period (days)", 7, 90, 30)
                predictions = self.predictor.predict(days)
                
                dates = [datetime.date.today() + timedelta(days=i) for i in range(1, days+1)]
                chart_data = pd.DataFrame({
                    "Date": dates,
                    "Predicted Savings": predictions
                })
                st.line_chart(chart_data.set_index("Date"))
        
        with tab3:
            st.write("### Unusual Activity")
            anomalies = self.detector.detect(self.wallet[self.username]["transactions"])
            if anomalies:
                st.warning("Potential anomalies detected:")
                st.dataframe(pd.DataFrame(anomalies))
            else:
                st.success("No unusual transactions found")

    def show_dashboard(self):
        st.subheader("📊 Financial Dashboard")
        
        # Summary cards
        cols = st.columns(4)
        summary = self.get_spending_summary()
        cols[0].metric("Balance", f"₹{self.wallet[self.username]['balance']:.2f}")
        cols[1].metric("Today", f"₹{summary['today']:.2f}")
        cols[2].metric("This Month", f"₹{summary['month']:.2f}")
        cols[3].metric("This Year", f"₹{summary['year']:.2f}")
        
        # Show total amount if set
        total_amount = self.wallet[self.username].get("total_amount", 0)
        if total_amount > 0:
            st.write(f"### Total Available Amount: ₹{total_amount:.2f}")
            remaining = total_amount - self.wallet[self.username]["balance"]
            st.progress(min(1.0, self.wallet[self.username]["balance"] / total_amount))
            st.write(f"Remaining: ₹{remaining:.2f} ({remaining/total_amount*100:.1f}%)")
        
        # Quick actions
        st.write("### Quick Actions")
        action_cols = st.columns(3)
        
        with action_cols[0]:
            with st.expander("➕ Add Savings"):
                self.show_add_savings_form()
        
        with action_cols[1]:
            with st.expander("➖ Record Expense"):
                self.show_spend_money_form()
        
        with action_cols[2]:
            with st.expander("📊 View History"):
                if not self.wallet[self.username]["transactions"]:
                    st.write("No transactions yet")
                else:
                    st.dataframe(pd.DataFrame(self.wallet[self.username]["transactions"]).head(5))
        
        # Monthly trends
        st.write("### Monthly Trends")
        monthly_data = self.get_monthly_summary()
        if not monthly_data.empty:
            st.bar_chart(monthly_data.set_index('Month')[['Savings', 'Spending']])
        else:
            st.write("No data available yet")

    def show_add_savings_form(self):
        """Interactive form for adding savings"""
        with st.form("add_savings_form"):
            amount = st.number_input("Amount (₹)", min_value=100.0, step=100.0)
            category = st.selectbox("Category", CATEGORIES)
            
            if st.form_submit_button("Add Savings"):
                self.add_saving(amount, category)
                st.success("Savings added successfully!")
                st.rerun()

    def show_spend_money_form(self):
        """Interactive form for spending"""
        with st.form("spend_money_form"):
            amount = st.number_input("Amount (₹)", min_value=1.0, step=100.0)
            description = st.text_input("Description")
            
            # Auto-categorization
            auto_category = self.categorizer.categorize(description) if description else "Other"
            category = st.selectbox(
                "Category",
                [c for c in CATEGORIES if c != "Savings"],
                index=[c for c in CATEGORIES if c != "Savings"].index(auto_category) 
                if auto_category in [c for c in CATEGORIES if c != "Savings"] else 0
            )
            
            if st.form_submit_button("Record Expense"):
                self.spend_money(amount, category, description)
                st.success("Expense recorded successfully!")
                st.rerun()

    def get_monthly_summary(self):
        """Get monthly summary data for visualizations"""
        df = pd.DataFrame(self.wallet[self.username]["transactions"])
        if df.empty:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'])
        df['Month'] = df['date'].dt.to_period('M')
        
        # Group by month and transaction type, sum amounts
        monthly = df.groupby(['Month', 'type'])['amount'].sum().unstack(fill_value=0)
        
        # Rename columns if they exist
        if 'Saving' in monthly.columns:
            monthly = monthly.rename(columns={'Saving': 'Savings'})
        if 'Spend' in monthly.columns:
            monthly = monthly.rename(columns={'Spend': 'Spending'})
            monthly['Spending'] = monthly['Spending'].abs()
        else:
            monthly['Spending'] = 0  # Add Spending column if it doesn't exist
        
        # Ensure Savings column exists
        if 'Savings' not in monthly.columns:
            monthly['Savings'] = 0
            
        return monthly.reset_index()

# Login Page
def login_page():
    st.title("💰 AI-Powered Savings Tracker")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Login"):
                users = load_data("users.json")
                if username in users and users[username] == hash_password(password):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            
            if st.form_submit_button("Register"):
                users = load_data("users.json")
                if not new_user or not new_pass:
                    st.error("Please enter username and password")
                elif new_user in users:
                    st.error("Username already exists")
                elif new_pass != confirm_pass:
                    st.error("Passwords don't match")
                else:
                    users[new_user] = hash_password(new_pass)
                    save_data(users, "users.json")
                    
                    # Initialize user data
                    wallet = load_data("wallet.json")
                    wallet[new_user] = {
                        "balance": 0.0,
                        "transactions": [],
                        "goals": {},
                        "total_amount": 0.0
                    }
                    save_data(wallet, "wallet.json")
                    
                    plans = load_data("spending_plans.json")
                    plans[new_user] = {
                        "daily": {},
                        "monthly": {cat: 0.0 for cat in CATEGORIES if cat != "Savings"}
                    }
                    save_data(plans, "spending_plans.json")
                    
                    st.success("Registration successful! Please login")

# Main App
def main():
    st.set_page_config(page_title="Savings Tracker", layout="wide")
    
    if "logged_in" not in st.session_state:
        login_page()
        return

    username = st.session_state["username"]
    tracker = SavingTracker(username)

    # Sidebar navigation
    st.sidebar.title(f"Welcome, {username}")
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Dashboard", "💸 Savings", "💳 Expenses", "📝 Budget", "📜 History", "🤖 AI Insights"],
        key="page"
    )
    
    st.sidebar.markdown("---")
    tracker.show_notifications()
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

    # Main content
    if page == "📊 Dashboard":
        tracker.show_dashboard()
    elif page == "💸 Savings":
        st.header("💸 Add Savings")
        tracker.show_add_savings_form()
    elif page == "💳 Expenses":
        st.header("💳 Record Expenses")
        tracker.show_spend_money_form()
    elif page == "📝 Budget":
        st.header("📝 Budget Planning")
        tracker.show_spending_plan()
    elif page == "📜 History":
        st.header("📜 Transaction History")
        if not tracker.wallet[username]["transactions"]:
            st.write("No transactions yet")
        else:
            st.dataframe(pd.DataFrame(tracker.wallet[username]["transactions"]))
    elif page == "🤖 AI Insights":
        tracker.show_ai_dashboard()

if __name__ == "__main__":
    main()