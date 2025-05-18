from flask import Flask, render_template, request
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to store rules
stored_rules = pd.DataFrame()

def frozenset_to_str(fset):
    return ', '.join(list(fset))

@app.route("/", methods=["GET", "POST"])
def upload():
    global stored_rules
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            df = pd.read_csv(filepath, header=None)

# Clean transactions: remove NaNs and convert to strings
            transactions = []
            for row in df.values.tolist():
                clean_row = [str(item) for item in row if pd.notna(item)]
                transactions.append(clean_row)

            te = TransactionEncoder()
            te_array = te.fit(transactions).transform(transactions)
            df_binary = pd.DataFrame(te_array, columns=te.columns_)


            # Apply Apriori
            frequent_items = apriori(df_binary, min_support=0.3, use_colnames=True)
            rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)

            # Convert frozensets to strings for better display
            rules['antecedents'] = rules['antecedents'].apply(frozenset_to_str)
            rules['consequents'] = rules['consequents'].apply(frozenset_to_str)

            stored_rules = rules

            return render_template("result.html", rules=rules.to_dict("records"))

    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    global stored_rules
    product = request.form.get("product", "")

    if stored_rules.empty:
        return "No rules available. Please upload data first."

    filtered = stored_rules[stored_rules['antecedents'].str.contains(product, case=False, na=False)]

    # Antecedents and consequents are already strings
    return render_template("result.html", rules=filtered.to_dict("records"))

if __name__ == "__main__":
    app.run(debug=True)
