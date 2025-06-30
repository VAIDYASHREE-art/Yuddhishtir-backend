import pandas as pd
import json
import os


# Static column name mapping for clean CSVs
column_map = {
    "procedure_name": "AB PM - JAY Procedure Name",
    "specialty": "Specialty",
    "cost_inr": "Procedure Price"
}

def build_rag_prompt(input_csv_path: str, output_path: str):
    try:
        df = pd.read_csv(input_csv_path, skiprows=1)
        df.columns = df.columns.str.replace('\n', ' ').str.strip() 
    except Exception as e:
        raise FileNotFoundError(f"❌ Couldn't read CSV file: {input_csv_path}\n{e}")

    # Get exact mapped names
    proc_col = column_map["procedure_name"]
    spec_col = column_map["specialty"]
    cost_col = column_map["cost_inr"]

    for col in [proc_col, spec_col, cost_col]:
        if col not in df.columns:
            raise ValueError(f"🚫 Column '{col}' not found in CSV headers: {df.columns.tolist()}")

    print("✅ Using columns:")
    print(f"   ➤ Procedure → {proc_col}")
    print(f"   ➤ Specialty → {spec_col}")
    print(f"   ➤ Cost      → {cost_col}")

    documents = []
    for idx, row in df.iterrows():
        try:
            text = f"Procedure: {row[proc_col]}, Specialty: {row[spec_col]}, Price: ₹{row[cost_col]}"
            doc = {
                "id": f"proc_{idx}",
                "text": text,
                "metadata": {
                    "procedure_name": row[proc_col],
                    "specialty": row[spec_col],
                    "cost_inr": row[cost_col]
                }
            }
            documents.append(doc)
        except Exception as row_err:
            print(f"⚠️ Skipping row {idx} due to error: {row_err}")
            continue

    rag_data = {
        "documents": documents,
        "total_count": len(documents)
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rag_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved RAG data to {output_path}")