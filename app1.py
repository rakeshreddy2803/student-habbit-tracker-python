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
import base64
from PIL import Image

# Initialize NLTK data for TextBlob
try:
    nltk_download('punkt')
    nltk_download('averaged_perceptron_tagger')
except:
    pass

# Define categories for savings
CATEGORIES = ["Savings", "Food", "Clothing", "Trip Savings", "Education", "Transport", "Entertainment"]

# Define profile fields
PROFILE_FIELDS = {
    "name": "Full Name",
    "email": "Email",
    "phone": "Phone Number",
    "currency": "Preferred Currency",
    "savings_goal": "Monthly Savings Goal",
    "notifications": "Enable Notifications"
}

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"
if 'form_active' not in st.session_state:
    st.session_state.form_active = False
if 'profile_edit_mode' not in st.session_state:
    st.session_state.profile_edit_mode = False

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
    for file in ["users.json", "wallet.json", "spending_plans.json", "profiles.json"]:
        try:
            with open(file, "r") as f:
                json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            with open(file, "w") as f:
                json.dump({}, f)

init_files()

# AI Model Classes (keeping these for potential future use, though we'll remove the insights section)
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
        self.profiles = load_data("profiles.json")
        self.users = load_data("users.json")
        self.predictor = SavingsPredictor()
        self.categorizer = ExpenseCategorizer()
        self.detector = AnomalyDetector()
        
        if username not in self.wallet:
            self.wallet[username] = {
                "balance": 0.0,
                "transactions": [],
                "goals": {},
                "total_amount": 0.0
            }
            save_data(self.wallet, "wallet.json")
            
        if username not in self.spending_plans:
            self.spending_plans[username] = {
                "daily": {},
                "monthly": {cat: 0.0 for cat in CATEGORIES if cat != "Savings"}
            }
            save_data(self.spending_plans, "spending_plans.json")
            
        if username not in self.profiles:
            self.profiles[username] = {
                "name": "",
                "email": "",
                "phone": "",
                "currency": "₹",
                "savings_goal": 0.0,
                "notifications": True,
                "created_at": str(datetime.datetime.now()),
                "profile_pic": None
            }
            save_data(self.profiles, "profiles.json")

    def get_user_by_email(self, email):
        """Find username by email"""
        for username, profile in self.profiles.items():
            if profile.get("email", "").lower() == email.lower():
                return username
        return None

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
            
            total_amount = self.wallet[self.username].get("total_amount", 0)
            if total_amount > 0:
                balance = self.wallet[self.username]["balance"]
                remaining = total_amount - balance
                st.metric("Total Available", f"₹{total_amount:.2f}", f"₹{remaining:.2f} remaining")
            
            st.write(random.choice(savings_quotes))

    def show_dashboard(self):
        st.subheader("📊 Financial Dashboard")
        
        total_amount = self.wallet[self.username].get("total_amount", 0.0)
        balance = self.wallet[self.username]["balance"]
        savings = sum(t['amount'] for t in self.wallet[self.username]['transactions'] if t['amount'] > 0)
        spending = abs(sum(t['amount'] for t in self.wallet[self.username]['transactions'] if t['amount'] < 0))
        
        cols = st.columns(3)
        cols[0].metric("Total Amount", f"₹{total_amount:.2f}")
        cols[1].metric("Current Balance", f"₹{balance:.2f}")
        cols[2].metric("Total Savings", f"₹{savings:.2f}")
        
        if total_amount > 0:
            st.write(f"### Budget Utilization")
            remaining = total_amount - balance
            st.progress(min(1.0, balance / total_amount))
            st.write(f"Remaining: ₹{remaining:.2f} ({remaining/total_amount*100:.1f}%)")
        
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
        
        st.write("### Monthly Trends")
        monthly_data = self.get_monthly_summary()
        if not monthly_data.empty:
            st.bar_chart(monthly_data.set_index('Month')[['Savings', 'Spending']])
        else:
            st.write("No data available yet")

    def show_add_savings_form(self):
        with st.form("add_savings_form"):
            amount = st.number_input("Amount (₹)", min_value=100.0, step=100.0)
            category = st.selectbox("Category", CATEGORIES)
            
            if st.form_submit_button("Add Savings"):
                self.add_saving(amount, category)
                st.success("Savings added successfully!")
                st.rerun()

    def show_spend_money_form(self):
        with st.form("spend_money_form"):
            amount = st.number_input("Amount (₹)", min_value=1.0, step=100.0)
            description = st.text_input("Description")
            
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
        df = pd.DataFrame(self.wallet[self.username]["transactions"])
        if df.empty:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'])
        df['Month'] = df['date'].dt.to_period('M')
        
        monthly = df.groupby(['Month', 'type'])['amount'].sum().unstack(fill_value=0)
        
        if 'Saving' in monthly.columns:
            monthly = monthly.rename(columns={'Saving': 'Savings'})
        if 'Spend' in monthly.columns:
            monthly = monthly.rename(columns={'Spend': 'Spending'})
            monthly['Spending'] = monthly['Spending'].abs()
        else:
            monthly['Spending'] = 0
        
        if 'Savings' not in monthly.columns:
            monthly['Savings'] = 0
            
        return monthly.reset_index()

    def show_profile(self):
        st.header("👤 Profile Settings")
        
        col_img, col_info = st.columns([1, 3])
        
        with col_img:
            st.subheader("Profile Picture")
            if self.profiles[self.username].get("profile_pic"):
                try:
                    img_data = base64.b64decode(self.profiles[self.username]["profile_pic"])
                    st.image(img_data, width=150)
                except:
                    st.image("https://via.placeholder.com/150", width=150)
            else:
                st.image("https://via.placeholder.com/150", width=150)
            
            if st.session_state.profile_edit_mode:
                uploaded_file = st.file_uploader("Upload new photo", type=["jpg", "png", "jpeg"])
                if uploaded_file is not None:
                    img_data = uploaded_file.getvalue()
                    self.profiles[self.username]["profile_pic"] = base64.b64encode(img_data).decode()
                    save_data(self.profiles, "profiles.json")
                    st.success("Profile picture updated!")
                    st.rerun()

        with col_info:
            if st.session_state.profile_edit_mode:
                with st.form("profile_form"):
                    updated_profile = {}
                    
                    st.subheader("Personal Information")
                    updated_profile["name"] = st.text_input(
                        "Full Name",
                        value=self.profiles[self.username].get("name", "")
                    )
                    
                    updated_profile["email"] = st.text_input(
                        "Email Address",
                        value=self.profiles[self.username].get("email", "")
                    )
                    
                    updated_profile["phone"] = st.text_input(
                        "Phone Number",
                        value=self.profiles[self.username].get("phone", "")
                    )
                    
                    st.subheader("Preferences")
                    currency_options = ["₹", "$", "€", "£"]
                    current_currency = self.profiles[self.username].get("currency", "₹")
                    try:
                        current_index = currency_options.index(current_currency)
                    except ValueError:
                        current_index = 0
                    
                    updated_profile["currency"] = st.selectbox(
                        "Preferred Currency",
                        currency_options,
                        index=current_index
                    )
                    
                    updated_profile["savings_goal"] = st.number_input(
                        "Monthly Savings Goal",
                        min_value=0.0,
                        value=float(self.profiles[self.username].get("savings_goal", 0.0)),
                        step=100.0
                    )
                    
                    updated_profile["notifications"] = st.toggle(
                        "Enable Notifications",
                        value=self.profiles[self.username].get("notifications", True)
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("💾 Save Changes"):
                            if updated_profile["email"]:
                                for username, profile in self.profiles.items():
                                    if username != self.username and profile.get("email", "").lower() == updated_profile["email"].lower():
                                        st.error("This email is already registered to another account")
                                        return
                            
                            self.profiles[self.username].update(updated_profile)
                            save_data(self.profiles, "profiles.json")
                            st.session_state.profile_edit_mode = False
                            st.success("Profile updated successfully!")
                            st.balloons()
                            st.rerun()
                    with col2:
                        if st.form_submit_button("❌ Cancel"):
                            st.session_state.profile_edit_mode = False
                            st.rerun()
            else:
                profile = self.profiles[self.username]
                st.subheader("Personal Information")
                
                st.markdown(f"""
                **Name:** {profile.get("name", "Not set")}  
                **Email:** {profile.get("email", "Not set")}  
                **Phone:** {profile.get("phone", "Not set")}  
                **Currency:** {profile.get("currency", "₹")}  
                **Monthly Goal:** {profile.get("currency", "₹")}{profile.get("savings_goal", 0.0):.2f}  
                **Notifications:** {"✅ Enabled" if profile.get("notifications", True) else "❌ Disabled"}
                """)
                
                if st.button("✏️ Edit Profile", key="edit_profile_btn"):
                    st.session_state.profile_edit_mode = True
                    st.rerun()
        
        st.subheader("Account Statistics")
        
        try:
            created_at = datetime.datetime.strptime(
                self.profiles[self.username].get("created_at", str(datetime.datetime.now())),
                "%Y-%m-%d %H:%M:%S.%f"
            )
            account_age = (datetime.datetime.now() - created_at).days
        except:
            account_age = 0
        
        transactions = self.wallet[self.username]["transactions"]
        total_transactions = len(transactions)
        monthly_transactions = len([t for t in transactions if datetime.datetime.strptime(t["date"], "%Y-%m-%d").month == datetime.datetime.now().month])
        
        savings_goal = self.profiles[self.username].get("savings_goal", 0.0)
        current_savings = self.wallet[self.username]["balance"]
        savings_progress = min(1.0, current_savings / savings_goal) if savings_goal > 0 else 0
        
        st.metric("Account Age", f"{account_age} days")
        st.metric("Total Transactions", total_transactions)
        st.metric("Monthly Transactions", monthly_transactions)
        
        st.write("### Savings Progress")
        st.progress(savings_progress)
        st.write(f"Current: {self.profiles[self.username].get('currency', '₹')}{current_savings:.2f} / Goal: {self.profiles[self.username].get('currency', '₹')}{savings_goal:.2f}")
        
        st.write("### Recent Transactions")
        if transactions:
            recent_transactions = pd.DataFrame(transactions[-5:])
            st.dataframe(recent_transactions)
        else:
            st.write("No recent transactions")

# Login Page
def login_page():
    st.title("💰 Finovatex - Smart Finance Manager")
    
    tab1, tab2, tab3 = st.tabs(["Login", "Register", "Forgot Password"])
    
    with tab1:
        with st.form("login_form"):
            st.subheader("Welcome Back!")
            login_id = st.text_input("Username or Email")
            password = st.text_input("Password", type="password")
            remember_me = st.checkbox("Remember me")
            
            if st.form_submit_button("Login"):
                users = load_data("users.json")
                profiles = load_data("profiles.json")
                
                if "@" in login_id:
                    username = None
                    for user, profile in profiles.items():
                        if profile.get("email", "").lower() == login_id.lower():
                            username = user
                            break
                    
                    if not username:
                        st.error("No account found with this email")
                        return
                else:
                    username = login_id
                
                if username in users and users[username] == hash_password(password):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    if remember_me:
                        pass
                    st.success("Login successful!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    with tab2:
        with st.form("register_form"):
            st.subheader("Create New Account")
            new_user = st.text_input("Username (required)")
            new_email = st.text_input("Email Address (required)")
            new_pass = st.text_input("New Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            
            if st.form_submit_button("Register"):
                users = load_data("users.json")
                profiles = load_data("profiles.json")
                
                errors = []
                if not new_user:
                    errors.append("Username is required")
                if not new_email:
                    errors.append("Email is required")
                elif "@" not in new_email:
                    errors.append("Invalid email format")
                if not new_pass:
                    errors.append("Password is required")
                elif new_pass != confirm_pass:
                    errors.append("Passwords don't match")
                
                if new_user in users:
                    errors.append("Username already exists")
                
                for profile in profiles.values():
                    if profile.get("email", "").lower() == new_email.lower():
                        errors.append("Email already registered")
                        break
                
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    users[new_user] = hash_password(new_pass)
                    save_data(users, "users.json")
                    
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
                    
                    profiles[new_user] = {
                        "name": "",
                        "email": new_email,
                        "phone": "",
                        "currency": "₹",
                        "savings_goal": 0.0,
                        "notifications": True,
                        "created_at": str(datetime.datetime.now()),
                        "profile_pic": None
                    }
                    save_data(profiles, "profiles.json")
                    
                    st.success("Registration successful! Please login")
                    st.balloons()
    
    with tab3:
        with st.form("forgot_password_form"):
            st.subheader("Reset Password")
            email = st.text_input("Enter your registered email")
            
            if st.form_submit_button("Send Reset Link"):
                profiles = load_data("profiles.json")
                username = None
                
                for user, profile in profiles.items():
                    if profile.get("email", "").lower() == email.lower():
                        username = user
                        break
                
                if username:
                    st.success(f"Password reset link sent to {email}")
                    st.info("For this demo, please contact support to reset your password")
                else:
                    st.error("No account found with this email")

# Main App
def main():
    st.set_page_config(
        page_title="Finovatex",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .stButton>button {
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    .metric {
        border-radius: 10px;
        padding: 15px;
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if "logged_in" not in st.session_state:
        login_page()
        return

    username = st.session_state["username"]
    tracker = SavingTracker(username)

    if "page" not in st.session_state:
        st.session_state.page = "📊 Dashboard"

    col1, col2, col3 = st.columns([5, 3, 1])
    
    with col1:
        st.title("💰 Finovatex")
    
    with col2:
        profile = tracker.profiles.get(username, {})
        name = profile.get("name", "")
        if name:
            st.markdown(f"### Welcome back, {name}!")
        else:
            st.markdown("### Welcome back!")
    
    with col3:
        if st.button("👤 Profile", key="profile_btn", use_container_width=True):
            st.session_state.page = "👤 Profile"
            st.rerun()

    with st.sidebar:
        st.title("Navigation")
        nav_options = {
            "📊 Dashboard": "Dashboard",
            "📝 Budget Planner": "Budget",
            "📜 Transaction History": "History"
        }
        
        for option, page in nav_options.items():
            if st.button(option, key=f"nav_{page}", use_container_width=True):
                st.session_state.page = option
                st.rerun()
        
        st.markdown("---")
        tracker.show_notifications()

    if st.session_state.page == "📊 Dashboard":
        with st.container():
            tracker.show_dashboard()
    elif st.session_state.page == "📝 Budget Planner":
        with st.container():
            tracker.show_spending_plan()
    elif st.session_state.page == "📜 Transaction History":
        with st.container():
            st.header("📜 Transaction History")
            if not tracker.wallet[username]["transactions"]:
                st.info("No transactions yet")
            else:
                # Fixed transaction history display
                df = pd.DataFrame(tracker.wallet[username]["transactions"])
                # Convert date strings to datetime for proper sorting
                try:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date', ascending=False)
                    # Format date for display
                    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                except:
                    st.warning("Could not parse transaction dates")
                
                # Display the dataframe with better formatting
                st.dataframe(
                    df,
                    column_config={
                        "amount": st.column_config.NumberColumn(
                            "Amount",
                            format="₹%.2f",
                        )
                    },
                    use_container_width=True,
                    hide_index=True
                )
    elif st.session_state.page == "👤 Profile":
        with st.container():
            tracker.show_profile()
            if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
                st.session_state.clear()
                st.success("Logged out successfully!")
                st.rerun()

if __name__ == "__main__":
    main()