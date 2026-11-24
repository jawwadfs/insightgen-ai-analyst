"""
Generates rich, realistic sample business datasets for InsightGen.
"""
import os
import random
import csv

os.makedirs("datasets", exist_ok=True)

# 1. Superstore Sales Dataset
def generate_superstore():
    regions = ["North", "South", "East", "West", "Central"]
    segments = ["Consumer", "Corporate", "Home Office"]
    categories = {
        "Technology": ["Phones", "Laptops", "Accessories", "Copiers"],
        "Furniture": ["Chairs", "Tables", "Bookcases", "Furnishings"],
        "Office Supplies": ["Paper", "Binders", "Storage", "Appliances", "Art"]
    }
    ship_modes = ["Standard Class", "Second Class", "First Class", "Same Day"]
    
    rows = []
    random.seed(42)
    order_id = 1001
    
    for _ in range(600):
        cat = random.choice(list(categories.keys()))
        sub_cat = random.choice(categories[cat])
        region = random.choice(regions)
        segment = random.choice(segments)
        ship_mode = random.choice(ship_modes)
        
        qty = random.randint(1, 10)
        unit_price = round(random.uniform(15, 650), 2)
        if cat == "Technology":
            unit_price = round(random.uniform(50, 1200), 2)
        
        discount = round(random.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]), 2)
        sales = round(qty * unit_price * (1 - discount), 2)
        
        # Realistic margin
        margin_pct = random.uniform(-0.15, 0.35) if discount > 0.2 else random.uniform(0.10, 0.45)
        profit = round(sales * margin_pct, 2)
        
        rows.append({
            "OrderID": f"ORD-{order_id}",
            "Region": region,
            "Segment": segment,
            "Category": cat,
            "SubCategory": sub_cat,
            "ShipMode": ship_mode,
            "Sales": sales,
            "Quantity": qty,
            "Discount": discount,
            "Profit": profit
        })
        order_id += 1

    with open("datasets/superstore_sales.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Generated superstore_sales.csv with 600 records.")

# 2. Customer Churn Dataset
def generate_customer_churn():
    contracts = ["Month-to-month", "One year", "Two year"]
    payments = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    internets = ["Fiber optic", "DSL", "No"]
    
    rows = []
    random.seed(101)
    
    for i in range(1, 501):
        tenure = random.randint(1, 72)
        contract = random.choice(contracts)
        payment = random.choice(payments)
        internet = random.choice(internets)
        
        monthly = round(random.uniform(25.0, 115.0), 2)
        total = round(monthly * tenure * random.uniform(0.95, 1.05), 2)
        tech_support = random.choice(["Yes", "No"]) if internet != "No" else "No internet"
        streaming = random.choice(["Yes", "No"]) if internet != "No" else "No internet"
        
        # Churn probability higher if month-to-month and fiber optic and no tech support
        churn_prob = 0.15
        if contract == "Month-to-month":
            churn_prob += 0.30
        if tech_support == "No":
            churn_prob += 0.15
        if tenure < 12:
            churn_prob += 0.20
            
        churned = "Yes" if random.random() < churn_prob else "No"
        
        rows.append({
            "CustomerID": f"CUST-{1000 + i}",
            "TenureMonths": tenure,
            "Contract": contract,
            "PaymentMethod": payment,
            "InternetService": internet,
            "TechSupport": tech_support,
            "StreamingTV": streaming,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": churned
        })
        
    with open("datasets/customer_churn.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Generated customer_churn.csv with 500 records.")

# 3. Employee Attrition Dataset
def generate_employee_attrition():
    departments = ["Sales", "Research & Development", "Human Resources"]
    roles = {
        "Sales": ["Sales Executive", "Sales Representative", "Manager"],
        "Research & Development": ["Research Scientist", "Software Engineer", "Laboratory Technician"],
        "Human Resources": ["HR Specialist", "Recruiter", "HR Manager"]
    }
    
    rows = []
    random.seed(202)
    
    for i in range(1, 451):
        dept = random.choice(departments)
        role = random.choice(roles[dept])
        age = random.randint(22, 58)
        years_at_company = min(random.randint(1, 20), age - 21)
        overtime = random.choice(["Yes", "No"])
        satisfaction = random.randint(1, 4) # 1=Low, 4=Very High
        work_life_balance = random.randint(1, 4)
        
        base_salary = 35000 if role in ["Sales Representative", "Laboratory Technician"] else 65000
        if "Manager" in role:
            base_salary = 110000
        salary = int(base_salary + (years_at_company * 2500) + random.randint(-5000, 15000))
        
        # Attrition logic
        attrition_prob = 0.10
        if overtime == "Yes":
            attrition_prob += 0.25
        if satisfaction <= 2:
            attrition_prob += 0.25
        if work_life_balance == 1:
            attrition_prob += 0.20
            
        attrition = "Yes" if random.random() < attrition_prob else "No"
        
        rows.append({
            "EmployeeID": f"EMP-{5000 + i}",
            "Age": age,
            "Department": dept,
            "JobRole": role,
            "MonthlyIncome": round(salary / 12, 2),
            "YearsAtCompany": years_at_company,
            "OverTime": overtime,
            "JobSatisfaction": satisfaction,
            "WorkLifeBalance": work_life_balance,
            "Attrition": attrition
        })
        
    with open("datasets/employee_attrition.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Generated employee_attrition.csv with 450 records.")

if __name__ == "__main__":
    generate_superstore()
    generate_customer_churn()
    generate_employee_attrition()
