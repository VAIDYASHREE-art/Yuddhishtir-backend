from yuddhishitir.preprocessor import build_rag_data

input_csv = "processed_data/HBP_split_part1_of_4.csv"
output_path = "processed_data/aiims_rag_data.json"

build_rag_data(input_csv, output_path)